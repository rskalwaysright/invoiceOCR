"""
Fuzzy Product Catalog Matcher for OCR Post-Correction.

Uses RapidFuzz to match noisy extracted receipt text against a canonical
product catalog, correcting OCR corruptions like 'M1LK 1GAL' -> 'Whole Milk 1 Gallon'.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from rapidfuzz import fuzz, process


class CatalogMatcher:
    """Matches raw OCR line item descriptions against a product catalog."""

    # Common OCR character substitutions in supermarket receipts
    OCR_CHAR_MAP = {
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B",
        "@": "A",
        "$": "S",
        "|": "I",
    }

    def __init__(self, catalog_path: Optional[str] = None):
        self.catalog: List[Dict[str, Any]] = []
        self.lookup_entries: List[Dict[str, Any]] = []  # Flat list of searchable names/aliases

        if catalog_path and os.path.exists(catalog_path):
            self.load_catalog(catalog_path)

    def load_catalog(self, catalog_path: str) -> None:
        """Load product catalog from a JSON file."""
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.catalog = data.get("products", [])
        self.lookup_entries = []

        for prod in self.catalog:
            # Canonical name
            self.lookup_entries.append({
                "query_text": prod["name"].upper(),
                "product": prod,
                "is_alias": False,
            })
            # Aliases
            for alias in prod.get("aliases", []):
                self.lookup_entries.append({
                    "query_text": alias.upper(),
                    "product": prod,
                    "is_alias": True,
                })

    def match_item(
        self,
        raw_description: str,
        threshold: float = 65.0,
    ) -> Dict[str, Any]:
        """
        Fuzzy match a single item description against the catalog.
        
        Returns:
            {
                "original_description": str,
                "matched": bool,
                "sku": Optional[str],
                "canonical_name": Optional[str],
                "category": Optional[str],
                "typical_price": Optional[float],
                "confidence": float,
                "matched_alias": Optional[str],
                "corrected": bool
            }
        """
        if not self.lookup_entries:
            return {
                "original_description": raw_description,
                "matched": False,
                "confidence": 0.0,
                "corrected": False,
                "reason": "Catalog is empty",
            }

        cleaned = self._clean_ocr_text(raw_description)
        normalized = self._apply_ocr_normalization(cleaned)

        best_match = None
        best_score = 0.0

        for entry in self.lookup_entries:
            candidate = entry["query_text"]

            # Compute multiple fuzz metrics
            score_token_sort = fuzz.token_sort_ratio(cleaned, candidate)
            score_token_set = fuzz.token_set_ratio(cleaned, candidate)
            score_norm = fuzz.token_sort_ratio(normalized, candidate)
            score_partial = fuzz.partial_ratio(cleaned, candidate)

            # Combined weighted score
            score = max(score_token_sort, score_token_set, score_norm, score_partial * 0.9)

            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= threshold:
            prod = best_match["product"]
            is_corrected = cleaned.upper() != prod["name"].upper()
            return {
                "original_description": raw_description,
                "matched": True,
                "sku": prod.get("sku"),
                "canonical_name": prod.get("name"),
                "category": prod.get("category"),
                "typical_price": prod.get("typical_price"),
                "confidence": round(best_score, 1),
                "matched_alias": best_match["query_text"],
                "corrected": is_corrected,
            }

        return {
            "original_description": raw_description,
            "matched": False,
            "sku": None,
            "canonical_name": raw_description,
            "category": "Uncategorized",
            "typical_price": None,
            "confidence": round(best_score, 1),
            "matched_alias": None,
            "corrected": False,
        }

    def match_all(
        self,
        line_items: List[Dict[str, Any]],
        threshold: float = 65.0,
    ) -> List[Dict[str, Any]]:
        """Enhance a list of parsed line items with catalog matches and corrected names."""
        enhanced_items = []
        for item in line_items:
            desc = item.get("description", "")
            match_res = self.match_item(desc, threshold=threshold)

            # Merge matched catalog fields into line item
            enriched = dict(item)
            enriched["catalog_match"] = match_res
            if match_res["matched"]:
                enriched["matched_name"] = match_res["canonical_name"]
                enriched["sku"] = match_res["sku"]
                enriched["category"] = match_res["category"]
            else:
                enriched["matched_name"] = desc
                enriched["sku"] = None
                enriched["category"] = "Uncategorized"

            enhanced_items.append(enriched)
        return enhanced_items

    def _clean_ocr_text(self, text: str) -> str:
        """Strip non-alphanumeric noise symbols from OCR text."""
        cleaned = re.sub(r"[^A-Za-z0-9\s#%&]", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip().upper()

    def _apply_ocr_normalization(self, text: str) -> str:
        """Replace likely digit-to-letter OCR errors for fuzzy match boosting."""
        chars = [self.OCR_CHAR_MAP.get(c, c) for c in text]
        return "".join(chars)
