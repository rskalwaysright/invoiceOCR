"""
Receipt & Invoice Line-Item Extraction Engine
"""

__version__ = "1.0.0"

from src.preprocess import preprocess_receipt
from src.ocr import OCREngine
from src.parser import ReceiptParser
from src.matcher import CatalogMatcher
from src.evaluate import PipelineEvaluator
from src.parallel_engine import ParallelReceiptEngine

__all__ = [
    "preprocess_receipt",
    "OCREngine",
    "ReceiptParser",
    "CatalogMatcher",
    "PipelineEvaluator",
    "ParallelReceiptEngine",
]
