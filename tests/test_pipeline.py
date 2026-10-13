"""
Comprehensive Unit & Integration Test Suite for Receipt Extraction Pipeline.
"""

import os
import unittest
import numpy as np
import cv2

from src.preprocess import preprocess_receipt, detect_skew_angle, correct_lighting
from src.parser import ReceiptParser
from src.matcher import CatalogMatcher
from src.parallel_engine import ParallelReceiptEngine


class TestReceiptPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = ReceiptParser()
        cls.matcher = CatalogMatcher("catalog.json") if os.path.exists("catalog.json") else None

    def test_parser_basic_items(self):
        sample_text = """
METRO FRESH MARKET
1042 MARKET ST, SAN FRANCISCO, CA
DATE: 10/11/2026 TIME: 12:35 PM
----------------------------------------
SEA SALT POTATO CHIPS 8OZ    $  3.99
GRADE A LARGE EGGS 12CT      $  3.49
WHOLE MILK 1GAL              $  4.29
----------------------------------------
SUBTOTAL                     $ 11.77
TAX (8.25%)                  $  0.97
TOTAL                        $ 12.74
PAYMENT METHOD: VISA **** 4812
THANK YOU FOR SHOPPING WITH US!
"""
        res = self.parser.parse(sample_text)
        self.assertEqual(res["merchant_name"], "METRO FRESH MARKET")
        self.assertEqual(res["date"], "10/11/2026")
        self.assertEqual(res["time"], "12:35 PM")
        self.assertEqual(len(res["line_items"]), 3)
        
        # Verify first item clean description (no date/time leakage)
        first_desc = res["line_items"][0]["description"]
        self.assertNotIn("DATE", first_desc)
        self.assertNotIn("TIME", first_desc)
        self.assertEqual(res["line_items"][0]["total_price"], 3.99)

        # Financial validation
        fin = res["financial_summary"]
        self.assertAlmostEqual(fin["calculated_items_sum"], 11.77, places=2)
        self.assertAlmostEqual(fin["parsed_total"], 12.74, places=2)
        self.assertAlmostEqual(fin["discrepancy"], 0.0, places=2)

    def test_parser_multiline_items(self):
        """Test items where quantity/name is on one line and price is on the following line."""
        multiline_text = """
ORGANIC BANANAS
$ 1.99
SOURDOUGH BREAD LOAF
$ 4.49
SUBTOTAL $ 6.48
TOTAL $ 6.48
"""
        res = self.parser.parse(multiline_text)
        self.assertEqual(len(res["line_items"]), 2)
        self.assertEqual(res["line_items"][0]["total_price"], 1.99)
        self.assertEqual(res["line_items"][1]["total_price"], 4.49)

    def test_catalog_matcher(self):
        if not self.matcher:
            self.skipTest("catalog.json not present")

        # Exact alias match
        m1 = self.matcher.match_item("WHOLE MILK 1GAL")
        self.assertTrue(m1["matched"])
        self.assertEqual(m1["canonical_name"], "Whole Milk 1 Gallon")
        self.assertEqual(m1["sku"], "DAIRY-001")

        # Fuzzy OCR corrupted match
        m2 = self.matcher.match_item("WHL MILK 1 GAL")
        self.assertTrue(m2["matched"])
        self.assertEqual(m2["canonical_name"], "Whole Milk 1 Gallon")

    def test_preprocessing_output_format(self):
        # Create a synthetic white receipt test image
        img = np.ones((400, 300, 3), dtype=np.uint8) * 240
        cv2.putText(img, "TEST RECEIPT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
        
        res = preprocess_receipt(img, return_mode="binary")
        self.assertIn("final", res)
        self.assertIn("skew_angle", res)
        self.assertIn("deskewed", res)
        self.assertIn("binary", res)
        self.assertEqual(res["final"].shape[:2], (400, 300))

    def test_parallel_engine_batch(self):
        samples = [
            "data/samples/sample_01_clean.png",
            "data/samples/sample_02_skewed.png",
        ]
        existing = [p for p in samples if os.path.exists(p)]
        if not existing:
            self.skipTest("Sample images not generated")

        engine = ParallelReceiptEngine(default_workers=2)
        batch_res = engine.process_batch_parallel(existing, num_workers=2)
        self.assertEqual(batch_res["total_receipts"], len(existing))
        self.assertEqual(len(batch_res["results"]), len(existing))
        for item in batch_res["results"]:
            self.assertEqual(item["status"], "success")
            self.assertIn("line_items", item["result"])


if __name__ == "__main__":
    unittest.main()
