"""
Parallel Computing Engine for Receipt & Invoice Processing.

Provides high-performance multi-core parallel processing for large-scale
receipt OCR ingestion, batch deskewing, parsing, and catalog normalization.

Includes:
1. ProcessPoolExecutor-based parallel batch worker pool
2. Dynamic load distribution across available CPU cores
3. Speedup benchmarking (T_1 vs T_p)
4. Multi-core efficiency metrics (E = S / p)
5. Amdahl's Law theoretical scaling and bottleneck analysis
"""

import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple

from src.preprocess import preprocess_receipt
from src.ocr import OCREngine
from src.parser import ReceiptParser
from src.matcher import CatalogMatcher


def _receipt_worker_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Isolated worker task executed across separate OS process threads.
    Guarantees thread-safety and avoids Python GIL contention.
    """
    image_path = payload["image_path"]
    enable_preprocess = payload.get("enable_preprocess", True)
    catalog_path = payload.get("catalog_path", "catalog.json")
    ocr_backend = payload.get("ocr_backend", "auto")

    t_start = time.perf_counter()

    try:
        ocr = OCREngine(preferred_backend=ocr_backend)
        parser = ReceiptParser()
        matcher = CatalogMatcher(catalog_path) if os.path.exists(catalog_path) else None

        skew_angle = 0.0
        if enable_preprocess:
            return_mode = "enhanced_grayscale" if ocr.backend == "rapidocr" else "binary"
            prep_res = preprocess_receipt(image_path, return_mode=return_mode)
            image_to_ocr = prep_res["final"]
            skew_angle = prep_res["skew_angle"]
        else:
            image_to_ocr = image_path

        extracted_text = ocr.extract_text(image_to_ocr)
        parsed = parser.parse(extracted_text)

        if matcher:
            parsed["line_items"] = matcher.match_all(parsed["line_items"])

        t_elapsed = time.perf_counter() - t_start

        parsed["metadata"] = {
            "source_image": os.path.basename(image_path),
            "image_path": image_path,
            "preprocessed": enable_preprocess,
            "skew_angle_deg": skew_angle,
            "ocr_backend": ocr.backend,
            "worker_pid": os.getpid(),
            "processing_time_sec": round(t_elapsed, 4),
        }
        return {"status": "success", "result": parsed, "image_path": image_path}

    except Exception as e:
        t_elapsed = time.perf_counter() - t_start
        return {
            "status": "error",
            "error": str(e),
            "image_path": image_path,
            "processing_time_sec": round(t_elapsed, 4),
        }


class ParallelReceiptEngine:
    """High-performance multi-process pipeline for batch receipt extraction."""

    def __init__(self, default_workers: Optional[int] = None):
        self.cpu_cores = os.cpu_count() or 4
        self.max_workers = default_workers or min(self.cpu_cores, 8)

    def process_batch_parallel(
        self,
        image_paths: List[str],
        num_workers: Optional[int] = None,
        enable_preprocess: bool = True,
        catalog_path: str = "catalog.json",
        ocr_backend: str = "auto",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a list of receipt images concurrently using a pool of worker processes.
        """
        workers = num_workers or self.max_workers
        total = len(image_paths)
        if total == 0:
            return {"results": [], "total_time_sec": 0.0, "workers": workers, "throughput_fps": 0.0}

        payloads = [
            {
                "image_path": p,
                "enable_preprocess": enable_preprocess,
                "catalog_path": catalog_path,
                "ocr_backend": ocr_backend,
            }
            for p in image_paths
        ]

        t_start = time.perf_counter()
        results = [None] * total
        completed_count = 0

        # Execute using ProcessPoolExecutor for true multi-core parallel speedup
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_receipt_worker_task, payloads[i]): i
                for i in range(total)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                results[idx] = res
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total, res)

        total_wall_time = time.perf_counter() - t_start
        throughput = total / max(total_wall_time, 1e-6)

        return {
            "results": results,
            "total_receipts": total,
            "workers_used": workers,
            "wall_clock_time_sec": round(total_wall_time, 3),
            "throughput_receipts_per_sec": round(throughput, 2),
            "average_time_per_receipt_sec": round(total_wall_time / total, 3),
        }

    def benchmark_scaling(
        self,
        image_paths: List[str],
        worker_counts: Optional[List[int]] = None,
        enable_preprocess: bool = True,
        catalog_path: str = "catalog.json",
        ocr_backend: str = "auto",
    ) -> Dict[str, Any]:
        """
        Perform empirical parallel scaling analysis (Sequential vs Multi-Worker).
        Evaluates:
        - Wall-clock time T_p
        - Speedup Factor S_p = T_1 / T_p
        - Parallel Efficiency E_p = S_p / p
        - Amdahl's Law parallel fraction (P) estimation
        """
        if not image_paths:
            raise ValueError("No images provided for benchmarking.")

        if worker_counts is None:
            max_c = min(self.cpu_cores, 8)
            worker_counts = [1]
            for c in [2, 4, 8]:
                if c <= max_c and c not in worker_counts:
                    worker_counts.append(c)

        scaling_runs = []
        baseline_time = None

        for p in worker_counts:
            batch_summary = self.process_batch_parallel(
                image_paths=image_paths,
                num_workers=p,
                enable_preprocess=enable_preprocess,
                catalog_path=catalog_path,
                ocr_backend=ocr_backend,
            )
            wall_time = batch_summary["wall_clock_time_sec"]

            if p == 1 or baseline_time is None:
                baseline_time = wall_time

            speedup = round(baseline_time / max(wall_time, 1e-6), 2)
            efficiency = round((speedup / p) * 100, 1)

            scaling_runs.append({
                "workers": p,
                "wall_time_sec": wall_time,
                "speedup": speedup,
                "efficiency_pct": efficiency,
                "throughput_rps": batch_summary["throughput_receipts_per_sec"],
            })

        # Calculate estimated parallel fraction P using Amdahl's formula on max worker run
        amdahl_p = 0.85
        if len(scaling_runs) > 1:
            highest_run = scaling_runs[-1]
            p_val = highest_run["workers"]
            s_val = highest_run["speedup"]
            if p_val > 1 and s_val > 1.0:
                amdahl_p = min(0.99, max(0.1, (p_val * (s_val - 1.0)) / ((p_val - 1.0) * s_val)))

        theoretical_max_speedup = round(1.0 / (1.0 - amdahl_p + 1e-6), 2)

        return {
            "total_images_tested": len(image_paths),
            "cpu_cores_available": self.cpu_cores,
            "baseline_sequential_time_sec": baseline_time,
            "scaling_results": scaling_runs,
            "estimated_parallel_fraction": round(amdahl_p, 3),
            "theoretical_max_speedup_amdahl": theoretical_max_speedup,
        }
