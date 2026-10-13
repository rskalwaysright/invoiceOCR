"""
Unified OCR Engine with Multi-Backend Support:
1. RapidOCR (Deep-learning ONNX runtime - zero system dependencies, fast, accurate)
2. Pytesseract / Tesseract OCR (Classic open-source OCR engine with auto-discovery on Windows/Linux)

Provides consistent extraction API:
- extract_text(): Returns formatted multiline string
- extract_data(): Returns token/word-level bounding boxes and confidence scores
- get_status(): Engine health check and available backends
"""

import os
import shutil
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Check backend availability
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


def find_tesseract_binary() -> Optional[str]:
    """Search system PATH and known Windows installation directories for tesseract.exe."""
    in_path = shutil.which("tesseract")
    if in_path:
        return in_path

    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in common_windows_paths:
        if os.path.isfile(path):
            return path

    custom = os.environ.get("TESSERACT_CMD")
    if custom and os.path.isfile(custom):
        return custom

    return None


class OCREngine:
    """Unified OCR Engine supporting both RapidOCR (ONNX) and Tesseract."""

    def __init__(self, preferred_backend: str = "auto", tesseract_cmd: Optional[str] = None):
        """
        Args:
            preferred_backend: 'auto', 'rapidocr', or 'tesseract'
            tesseract_cmd: Optional explicit path to tesseract.exe
        """
        self.tesseract_cmd = tesseract_cmd or find_tesseract_binary()
        self.tesseract_configured = False

        if PYTESSERACT_AVAILABLE:
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                self.tesseract_configured = True
            elif shutil.which("tesseract"):
                self.tesseract_configured = True

        self._rapidocr_instance = None
        if RAPIDOCR_AVAILABLE:
            try:
                self._rapidocr_instance = RapidOCR()
            except Exception as e:
                logger.warning(f"Could not initialize RapidOCR: {e}")

        # Determine active backend
        if preferred_backend == "tesseract" and self.tesseract_configured:
            self.backend = "tesseract"
        elif preferred_backend == "rapidocr" and self._rapidocr_instance:
            self.backend = "rapidocr"
        else:
            # Auto selection: prefer RapidOCR if available (no external exe needed), else Tesseract
            if self._rapidocr_instance is not None:
                self.backend = "rapidocr"
            elif self.tesseract_configured:
                self.backend = "tesseract"
            else:
                self.backend = "none"

    def get_status(self) -> Dict[str, Any]:
        """Check active OCR backend status and available alternatives."""
        return {
            "active_backend": self.backend,
            "ready": self.backend != "none",
            "rapidocr_available": RAPIDOCR_AVAILABLE and (self._rapidocr_instance is not None),
            "tesseract_available": self.tesseract_configured,
            "tesseract_binary_path": self.tesseract_cmd,
        }

    def extract_text(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        psm: int = 6,
        oem: int = 3,
    ) -> str:
        """Extract multi-line receipt text."""
        if self.backend == "rapidocr":
            return self._extract_text_rapidocr(image_input)
        elif self.backend == "tesseract":
            return self._extract_text_tesseract(image_input, psm=psm, oem=oem)
        else:
            raise RuntimeError(
                "No OCR backend available. Install rapidocr-onnxruntime or configure Tesseract."
            )

    def extract_data(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        psm: int = 6,
        oem: int = 3,
    ) -> List[Dict[str, Any]]:
        """Extract word/phrase level items with bounding boxes and confidence scores."""
        if self.backend == "rapidocr":
            return self._extract_data_rapidocr(image_input)
        elif self.backend == "tesseract":
            return self._extract_data_tesseract(image_input, psm=psm, oem=oem)
        else:
            raise RuntimeError("No OCR backend available.")

    def _extract_text_rapidocr(self, img_input: Union[np.ndarray, Image.Image, str]) -> str:
        """Extract lines using RapidOCR."""
        img_np = self._to_numpy(img_input)
        result, _ = self._rapidocr_instance(img_np)
        if not result:
            return ""

        # RapidOCR returns list of [box, text, score]
        lines = [item[1].strip() for item in result if item[1].strip()]
        return "\n".join(lines)

    def _extract_data_rapidocr(self, img_input: Union[np.ndarray, Image.Image, str]) -> List[Dict[str, Any]]:
        """Extract boxes and confidence using RapidOCR."""
        img_np = self._to_numpy(img_input)
        result, _ = self._rapidocr_instance(img_np)
        if not result:
            return []

        data_items = []
        for i, item in enumerate(result):
            box_points = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = item[1].strip()
            score = float(item[2])

            # Convert 4 points polygon to bounding rect [left, top, width, height]
            xs = [pt[0] for pt in box_points]
            ys = [pt[1] for pt in box_points]
            left, top = int(min(xs)), int(min(ys))
            width, height = int(max(xs) - left), int(max(ys) - top)

            data_items.append({
                "text": text,
                "confidence": round(score * 100.0, 1),
                "box": {"left": left, "top": top, "width": width, "height": height},
                "polygon": box_points,
                "line_num": i + 1,
            })
        return data_items

    def _extract_text_tesseract(self, img_input, psm=6, oem=3) -> str:
        pil_img = self._to_pil_image(img_input)
        config = f"--oem {oem} --psm {psm}"
        return pytesseract.image_to_string(pil_img, config=config)

    def _extract_data_tesseract(self, img_input, psm=6, oem=3) -> List[Dict[str, Any]]:
        pil_img = self._to_pil_image(img_input)
        config = f"--oem {oem} --psm {psm}"
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        results = []
        for i in range(len(data["text"])):
            w = data["text"][i].strip()
            conf = float(data["conf"][i])
            if w and conf >= 0:
                results.append({
                    "text": w,
                    "confidence": conf,
                    "box": {
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    },
                    "line_num": data["line_num"][i],
                })
        return results

    def _to_numpy(self, img_input: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        if isinstance(img_input, str):
            import cv2
            img = cv2.imread(img_input)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_input}")
            return img
        elif isinstance(img_input, Image.Image):
            return np.array(img_input)
        elif isinstance(img_input, np.ndarray):
            return img_input
        raise TypeError("Invalid image type.")

    def _to_pil_image(self, img_input: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        if isinstance(img_input, str):
            return Image.open(img_input)
        elif isinstance(img_input, np.ndarray):
            if len(img_input.shape) == 2:
                return Image.fromarray(img_input)
            else:
                rgb = img_input[:, :, ::-1] if img_input.shape[2] == 3 else img_input
                return Image.fromarray(rgb)
        elif isinstance(img_input, Image.Image):
            return img_input
        raise TypeError("Invalid image type.")
