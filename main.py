"""
Receipt & Invoice Line-Item Extraction Engine - Main CLI Application.

Usage examples:
  1. Process a single receipt image:
     python main.py --image data/samples/sample_01_clean.png

  2. Process single receipt and save structured JSON:
     python main.py --image data/samples/sample_02_skewed.png --output output.json

  3. Run comparative accuracy benchmark (with vs without preprocessing):
     python main.py --benchmark

  4. Generate synthetic test dataset:
     python main.py --generate-samples

  5. Process entire folder of receipts in batch:
     python main.py --batch data/samples --output-dir results/
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

from src.preprocess import preprocess_receipt
from src.ocr import OCREngine
from src.parser import ReceiptParser
from src.matcher import CatalogMatcher
from src.evaluate import PipelineEvaluator
from src.parallel_engine import ParallelReceiptEngine


def process_single_receipt(
    image_path: str,
    enable_preprocess: bool = True,
    catalog_path: str = "catalog.json",
    ocr_backend: str = "auto",
) -> Dict[str, Any]:
    """Process a single receipt image from end to end."""
    print(f"[*] Processing receipt: {image_path}")

    # Initialize components
    ocr = OCREngine(preferred_backend=ocr_backend)
    parser = ReceiptParser()
    matcher = CatalogMatcher(catalog_path) if os.path.exists(catalog_path) else None

    # Step 1: Preprocessing
    skew_angle = 0.0
    if enable_preprocess:
        return_mode = "enhanced_grayscale" if ocr.backend == "rapidocr" else "binary"
        prep_res = preprocess_receipt(image_path, return_mode=return_mode)
        image_to_ocr = prep_res["final"]
        skew_angle = prep_res["skew_angle"]
        print(f"[*] Preprocessing complete. Detected skew: {skew_angle:+.2f}°")
    else:
        image_to_ocr = image_path
        print("[*] Skipping image preprocessing (running raw).")

    # Step 2: OCR Extraction
    print(f"[*] Running OCR backend: {ocr.backend}...")
    extracted_text = ocr.extract_text(image_to_ocr)
    raw_lines = [l for l in extracted_text.splitlines() if l.strip()]
    print(f"[*] Extracted {len(raw_lines)} text lines from OCR.")

    # Step 3: Parsing into line items and financials
    parsed = parser.parse(extracted_text)

    # Step 4: Fuzzy matching & auto-correction with catalog
    if matcher:
        parsed["line_items"] = matcher.match_all(parsed["line_items"])
        corrected_count = sum(1 for it in parsed["line_items"] if it.get("catalog_match", {}).get("corrected"))
        print(f"[*] Catalog matching: {corrected_count} item(s) auto-corrected with catalog.")

    # Attach processing metadata
    parsed["metadata"] = {
        "source_image": os.path.basename(image_path),
        "preprocessed": enable_preprocess,
        "skew_angle_deg": skew_angle,
        "ocr_backend": ocr.backend,
    }

    return parsed


def run_benchmark_cli(catalog_path: str = "catalog.json"):
    """Run full comparative benchmark across test dataset."""
    samples_dir = "data/samples"
    gt_dir = "data/ground_truth"

    if not os.path.exists(samples_dir) or not os.path.exists(gt_dir):
        print("[!] Sample directory not found. Generating sample receipts first...")
        from data.generate_samples import generate_all_samples
        generate_all_samples()

    print("\n" + "=" * 65)
    print("      RECEIPT EXTRACTION ACCURACY BENCHMARK EVALUATION")
    print("=" * 65)
    evaluator = PipelineEvaluator(catalog_path=catalog_path)
    summary = evaluator.evaluate_dataset(samples_dir, gt_dir)

    if "error" in summary:
        print(f"[!] Error: {summary['error']}")
        return

    n = summary["total_receipts_evaluated"]
    w_out = summary["without_preprocessing"]
    w_in = summary["with_preprocessing"]
    delta = summary["delta"]

    print(f"\nTested on {n} receipts (data/samples/):")
    print("-" * 65)
    print(f"{'Metric':<32} | {'Raw (Without)':<14} | {'Preprocessed (With)':<14}")
    print("-" * 65)
    print(f"{'Character Error Rate (CER)':<32} | {w_out.get('mean_cer', 0.0):<14.4f} | {w_in.get('mean_cer', 0.0):<14.4f}")
    print(f"{'Line-Item F1-Score':<32} | {w_out.get('mean_line_item_f1', 0.0):<14.4f} | {w_in.get('mean_line_item_f1', 0.0):<14.4f}")
    print(f"{'Total Extraction Accuracy':<32} | {w_out.get('total_accuracy_pct', 0.0):<13.1f}% | {w_in.get('total_accuracy_pct', 0.0):<13.1f}%")
    print("-" * 65)
    print(f"[*] Performance Delta: Line-Item F1 Gain = {delta.get('f1_score_gain', 0):+.4f} ({delta.get('f1_relative_improvement_pct', 0):+.1f}%)")
    print("=" * 65 + "\n")


def run_parallel_benchmark_cli(catalog_path: str = "catalog.json", workers_list: Optional[List[int]] = None):
    """Run parallel scaling benchmark across available CPU cores (Speedup & Amdahl's Law)."""
    samples_dir = "data/samples"
    if not os.path.exists(samples_dir):
        from data.generate_samples import generate_all_samples
        generate_all_samples()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    print("\n" + "=" * 72)
    print("      PARALLEL COMPUTING SPEEDUP & SCALING BENCHMARK")
    print("=" * 72)
    print(f"[*] Total dataset receipts: {len(images)}")
    engine = ParallelReceiptEngine()
    print(f"[*] System CPU cores detected: {engine.cpu_cores}")

    bench_res = engine.benchmark_scaling(
        image_paths=images,
        worker_counts=workers_list,
        catalog_path=catalog_path,
    )

    print("-" * 72)
    print(f"{'Workers':<10} | {'Wall Time (s)':<14} | {'Speedup':<12} | {'Efficiency (%)':<15} | {'Throughput (rps)'}")
    print("-" * 72)
    for row in bench_res["scaling_results"]:
        w = row["workers"]
        t = row["wall_time_sec"]
        s = row["speedup"]
        e = row["efficiency_pct"]
        r = row["throughput_rps"]
        print(f"{w:<10} | {t:<14.2f} | {s:<12.2f}x | {e:<15.1f}% | {r:<.2f}")
    print("-" * 72)
    p_frac = bench_res["estimated_parallel_fraction"]
    max_s = bench_res["theoretical_max_speedup_amdahl"]
    print(f"[*] Amdahl's Law Analysis: Parallel Fraction P ~= {p_frac * 100:.1f}%")
    print(f"[*] Theoretical Max Speedup S_max ~= {max_s:.2f}x (independent of core count)")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Receipt & Invoice Line-Item Extraction Pipeline with Parallel Computing"
    )
    parser.add_argument("--image", type=str, help="Path to single receipt image to process")
    parser.add_argument("--output", type=str, help="Path to output JSON file (optional)")
    parser.add_argument("--batch", type=str, help="Directory containing receipt images to process in batch")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save batch JSON outputs")
    parser.add_argument("--parallel", action="store_true", help="Process batch receipts in parallel using multi-core worker pool")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for parallel batching (default: all cores)")
    parser.add_argument("--benchmark", action="store_true", help="Run comparative accuracy benchmark on data/samples")
    parser.add_argument("--parallel-benchmark", action="store_true", help="Run multi-core parallel speedup & scaling benchmark")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic test receipts")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--catalog", type=str, default="catalog.json", help="Path to product catalog JSON")
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "rapidocr", "tesseract"], help="OCR engine backend")

    args = parser.parse_args()

    if args.generate_samples:
        from data.generate_samples import generate_all_samples
        generate_all_samples()
        return

    if args.benchmark:
        run_benchmark_cli(catalog_path=args.catalog)
        return

    if args.parallel_benchmark:
        run_parallel_benchmark_cli(catalog_path=args.catalog)
        return

    if args.image:
        result = process_single_receipt(
            args.image,
            enable_preprocess=not args.no_preprocess,
            catalog_path=args.catalog,
            ocr_backend=args.engine,
        )

        json_str = json.dumps(result, indent=2)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"[+] Saved extracted JSON to: {args.output}")
        else:
            print("\n--- EXTRACTED STRUCTURED JSON ---")
            print(json_str)
        return

    if args.batch:
        os.makedirs(args.output_dir, exist_ok=True)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in os.listdir(args.batch) if os.path.splitext(f)[1].lower() in image_exts]
        full_paths = [os.path.join(args.batch, f) for f in files]
        print(f"[*] Found {len(files)} receipt images in {args.batch}")

        if args.parallel:
            parallel_engine = ParallelReceiptEngine(default_workers=args.workers)
            print(f"[*] Running parallel extraction with {parallel_engine.max_workers} worker processes...")
            batch_res = parallel_engine.process_batch_parallel(
                image_paths=full_paths,
                num_workers=args.workers,
                enable_preprocess=not args.no_preprocess,
                catalog_path=args.catalog,
                ocr_backend=args.engine,
            )
            for item in batch_res["results"]:
                if item and item.get("status") == "success":
                    base_name = os.path.splitext(os.path.basename(item["image_path"]))[0]
                    out_path = os.path.join(args.output_dir, f"{base_name}.json")
                    with open(out_path, "w", encoding="utf-8") as fp:
                        json.dump(item["result"], fp, indent=2)
                    print(f"[+] Saved {out_path}")
            print(f"[+] Parallel batch processing completed in {batch_res['wall_clock_time_sec']:.2f}s "
                  f"({batch_res['throughput_receipts_per_sec']:.2f} receipts/s)")
        else:
            for f in files:
                img_path = os.path.join(args.batch, f)
                res = process_single_receipt(
                    img_path,
                    enable_preprocess=not args.no_preprocess,
                    catalog_path=args.catalog,
                    ocr_backend=args.engine,
                )
                out_path = os.path.join(args.output_dir, f"{os.path.splitext(f)[0]}.json")
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(res, fp, indent=2)
                print(f"[+] Saved {out_path}")
        return

    # If no flags provided, show quick help & run default sample 1
    print("[*] No arguments provided. Processing sample receipt 1 as demo...")
    res = process_single_receipt("data/samples/sample_01_clean.png", catalog_path=args.catalog)
    print("\n--- SAMPLE EXTRACTED JSON ---")
    print(json.dumps(res, indent=2))
    print("\nTip: Run 'python main.py --help' to see all command options.")


if __name__ == "__main__":
    main()

