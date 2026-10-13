"""
Comparative Evaluation Benchmark for Receipt Line-Item Extraction.

Compares:
1. Raw OCR (Without Preprocessing)
2. Preprocessed OCR (Deskewing + Lighting Correction + Adaptive Binarization)
3. Preprocessed OCR + Fuzzy Catalog Matching

Metrics Evaluated:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Line-Item Precision, Recall, and F1-Score
- Financial Total Accuracy (%)
- Catalog Item Match Accuracy (%)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from rapidfuzz.distance import Levenshtein

from src.preprocess import preprocess_receipt
from src.ocr import OCREngine
from src.parser import ReceiptParser
from src.matcher import CatalogMatcher

logger = logging.getLogger(__name__)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return min(1.0, dist / max(1, len(reference)))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = Levenshtein.distance(ref_words, hyp_words)
    return min(1.0, dist / max(1, len(ref_words)))


class PipelineEvaluator:
    """Evaluates and compares OCR pipeline with and without image preprocessing."""

    def __init__(self, catalog_path: Optional[str] = "catalog.json"):
        self.ocr_engine = OCREngine()
        self.parser = ReceiptParser()
        self.matcher = CatalogMatcher(catalog_path) if catalog_path else None

    def evaluate_receipt(
        self,
        image_path: str,
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single receipt under two modes:
        1. 'raw' (without preprocessing)
        2. 'preprocessed' (deskewed + lighting corrected + adaptive binarized)
        """
        results = {}

        # 1. Raw extraction (Without Preprocessing)
        try:
            raw_text_without = self.ocr_engine.extract_text(image_path)
            parsed_without = self.parser.parse(raw_text_without)
            metrics_without = self._compute_metrics(raw_text_without, parsed_without, ground_truth)
        except Exception as e:
            raw_text_without = ""
            parsed_without = {"line_items": [], "financial_summary": {}}
            metrics_without = {"cer": 1.0, "wer": 1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "error": str(e)}

        # 2. Preprocessed extraction (With Preprocessing)
        try:
            return_mode = "enhanced_grayscale" if self.ocr_engine.backend == "rapidocr" else "binary"
            prep_res = preprocess_receipt(image_path, return_mode=return_mode)
            preprocessed_img = prep_res["final"]
            raw_text_with = self.ocr_engine.extract_text(preprocessed_img)
            parsed_with = self.parser.parse(raw_text_with)

            # Apply fuzzy matching if matcher available
            if self.matcher:
                parsed_with["line_items"] = self.matcher.match_all(parsed_with["line_items"])

            metrics_with = self._compute_metrics(raw_text_with, parsed_with, ground_truth)
            metrics_with["skew_angle_detected"] = prep_res["skew_angle"]
        except Exception as e:
            raw_text_with = ""
            parsed_with = {"line_items": [], "financial_summary": {}}
            metrics_with = {"cer": 1.0, "wer": 1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "error": str(e)}

        return {
            "image_path": image_path,
            "without_preprocessing": {
                "metrics": metrics_without,
                "parsed": parsed_without,
                "raw_text": raw_text_without,
            },
            "with_preprocessing": {
                "metrics": metrics_with,
                "parsed": parsed_with,
                "raw_text": raw_text_with,
            },
        }

    def _compute_metrics(
        self,
        extracted_text: str,
        parsed_data: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute CER, WER, and line item precision/recall/F1 compared to ground truth."""
        # Text level accuracy
        gt_text = ground_truth.get("full_text", "")
        cer = compute_cer(gt_text, extracted_text) if gt_text else None
        wer = compute_wer(gt_text, extracted_text) if gt_text else None

        # Line item level accuracy
        gt_items = ground_truth.get("line_items", [])
        pred_items = parsed_data.get("line_items", [])

        # Match predicted items to ground truth items
        matched_gt = set()
        tp = 0
        for p in pred_items:
            p_price = p.get("total_price")
            p_desc = p.get("description", "").lower()
            for i, gt in enumerate(gt_items):
                if i in matched_gt:
                    continue
                gt_price = gt.get("total_price")
                gt_desc = gt.get("description", "").lower()

                # Price match within $0.05 and description similarity > 60%
                price_match = abs(p_price - gt_price) <= 0.05 if (p_price is not None and gt_price is not None) else False
                desc_sim = Levenshtein.normalized_similarity(p_desc, gt_desc) if (p_desc and gt_desc) else 0

                if price_match or (desc_sim >= 0.70):
                    matched_gt.add(i)
                    tp += 1
                    break

        precision = (tp / len(pred_items)) if pred_items else 0.0
        recall = (tp / len(gt_items)) if gt_items else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Total accuracy
        gt_total = ground_truth.get("financial_summary", {}).get("total")
        pred_total = parsed_data.get("financial_summary", {}).get("parsed_total")
        total_correct = False
        if gt_total is not None and pred_total is not None:
            total_correct = abs(gt_total - pred_total) <= 0.05

        return {
            "cer": round(cer, 4) if cer is not None else None,
            "wer": round(wer, 4) if wer is not None else None,
            "char_accuracy": round(1.0 - cer, 4) if cer is not None else None,
            "item_precision": round(precision, 4),
            "item_recall": round(recall, 4),
            "item_f1": round(f1, 4),
            "total_correct": total_correct,
            "predicted_items_count": len(pred_items),
            "ground_truth_items_count": len(gt_items),
        }

    def evaluate_dataset(
        self,
        samples_dir: str,
        ground_truth_dir: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a complete dataset of receipt images against ground truth JSON files.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f for f in os.listdir(samples_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        if not image_files:
            return {"error": f"No image files found in {samples_dir}"}

        detailed_results = []
        sum_f1_without, sum_f1_with = 0.0, 0.0
        sum_cer_without, sum_cer_with = 0.0, 0.0
        cer_count = 0
        correct_totals_without, correct_totals_with = 0, 0

        for img_name in sorted(image_files):
            img_path = os.path.join(samples_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            gt_path = os.path.join(ground_truth_dir, f"{base_name}.json")

            if not os.path.exists(gt_path):
                logger.warning(f"No ground truth JSON found for {img_name}, skipping.")
                continue

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            res = self.evaluate_receipt(img_path, gt_data)
            detailed_results.append(res)

            m_without = res["without_preprocessing"]["metrics"]
            m_with = res["with_preprocessing"]["metrics"]

            sum_f1_without += m_without.get("item_f1", 0.0)
            sum_f1_with += m_with.get("item_f1", 0.0)

            if m_without.get("cer") is not None and m_with.get("cer") is not None:
                sum_cer_without += m_without["cer"]
                sum_cer_with += m_with["cer"]
                cer_count += 1

            if m_without.get("total_correct"):
                correct_totals_without += 1
            if m_with.get("total_correct"):
                correct_totals_with += 1

        n = len(detailed_results)
        if n == 0:
            return {"error": "No matching images and ground truth files found."}

        avg_f1_without = round(sum_f1_without / n, 4)
        avg_f1_with = round(sum_f1_with / n, 4)
        avg_cer_without = round(sum_cer_without / max(1, cer_count), 4) if cer_count else None
        avg_cer_with = round(sum_cer_with / max(1, cer_count), 4) if cer_count else None

        f1_improvement = round(avg_f1_with - avg_f1_without, 4)
        f1_percent_gain = round((f1_improvement / max(0.01, avg_f1_without)) * 100, 1)

        summary = {
            "total_receipts_evaluated": n,
            "without_preprocessing": {
                "mean_line_item_f1": avg_f1_without,
                "mean_cer": avg_cer_without,
                "total_accuracy_pct": round((correct_totals_without / n) * 100, 1),
            },
            "with_preprocessing": {
                "mean_line_item_f1": avg_f1_with,
                "mean_cer": avg_cer_with,
                "total_accuracy_pct": round((correct_totals_with / n) * 100, 1),
            },
            "delta": {
                "f1_score_gain": f1_improvement,
                "f1_relative_improvement_pct": f1_percent_gain,
                "cer_reduction": round((avg_cer_without - avg_cer_with), 4) if (avg_cer_without and avg_cer_with) else None,
            },
            "detailed_results": detailed_results,
        }

        return summary
