"""
Image Preprocessing Pipeline for Receipt & Invoice OCR.

Key Operations:
1. Grayscale Conversion
2. Lighting & Shadow Correction (Morphological Background Division & CLAHE)
3. Noise Reduction (Bilateral & Gaussian Filtering)
4. Deskewing (Orientation Detection via Minimum Area Bounding Box / Hough Lines)
5. Adaptive Binarization / Thresholding (Adaptive Gaussian & Otsu)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Union


def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """Load an image from a filepath or return a copy if already an ndarray."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError("Input must be a filepath string or numpy ndarray.")


def correct_lighting(gray_image: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """
    Remove uneven illumination, camera flash hot-spots, and shadows.
    Uses morphological background division followed by CLAHE (Contrast-Limited Adaptive Histogram Equalization).
    """
    # Step 1: Estimate background illumination by aggressive morphological dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    bg_illumination = cv2.morphologyEx(gray_image, cv2.MORPH_DILATE, kernel)
    bg_illumination = cv2.medianBlur(bg_illumination, 21)

    # Step 2: Normalize image by dividing by the estimated background
    # Add epsilon to prevent division by zero
    diff = 255 - cv2.absdiff(gray_image, bg_illumination)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Step 3: Enhance local contrast of faint receipt text using CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(norm)

    return enhanced


def normalize_angle(angle: float) -> float:
    """Normalize any rotation angle into the range [-45.0, 45.0] degrees."""
    while angle <= -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    return float(angle)


def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect the tilt/skew angle (in degrees) of a receipt.
    
    Uses a hybrid approach:
    1. Paper boundary contour (dominant quad sheet on counter/table)
    2. Text-line horizontal orientation via morphological dilation
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    (h, w) = gray.shape[:2]
    total_area = h * w

    # Strategy 1: Check for receipt paper boundary against background
    _, thresh_paper = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        # If paper is a distinct region covering between 15% and 92% of the frame
        if 0.15 * total_area < area < 0.95 * total_area:
            rect = cv2.minAreaRect(largest)
            rw, rh = rect[1]
            ang = rect[2]
            if rw < rh:
                ang -= 90.0
            return normalize_angle(ang)

    # Strategy 2: Text-line horizontal orientation
    # Adaptive threshold to isolate dark text characters
    thresh_text = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    # Dilate horizontally to merge letters into continuous line blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh_text, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in contours:
        if cv2.contourArea(c) > 300:
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            ang = rect[2]
            if rw < rh:
                ang -= 90.0
            ang_norm = normalize_angle(ang)
            if abs(ang_norm) < 45.0:
                angles.append(ang_norm)

    if angles:
        return float(np.median(angles))

    return 0.0


def rotate_image(image: np.ndarray, angle: float, background_color: int = 255) -> np.ndarray:
    """
    Rotate the image around its center by the specified angle without cropping corners.
    """
    if abs(angle) < 0.1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    # Compute new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation component of the matrix
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    border_val = (background_color, background_color, background_color) if len(image.shape) == 3 else background_color
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return rotated


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect skew angle and deskew the image. Returns (deskewed_image, detected_angle).
    """
    angle = detect_skew_angle(image)
    deskewed = rotate_image(image, angle)
    return deskewed, angle


def binarize(gray_image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert grayscale image into crisp black-and-white binary image.
    Methods:
        - 'adaptive': Adaptive Gaussian thresholding (best for varied receipts)
        - 'otsu': Global Otsu thresholding
        - 'hybrid': Blended adaptive + Otsu mask
    """
    if method == "otsu":
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "hybrid":
        # Adaptive Gaussian
        adapt = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        # Otsu
        _, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Combine
        binary = cv2.bitwise_or(adapt, otsu)
    else:  # default adaptive
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
        )

    return binary


def denoise(gray_image: np.ndarray) -> np.ndarray:
    """
    Denoise while preserving sharp character contours.
    Uses bilateral filter to smooth paper background without blurring text edges.
    """
    return cv2.bilateralFilter(gray_image, d=7, sigmaColor=50, sigmaSpace=50)


def preprocess_receipt(
    image_input: Union[str, np.ndarray],
    enable_deskew: bool = True,
    enable_lighting_correction: bool = True,
    enable_denoise: bool = True,
    binarize_method: str = "adaptive",
    return_mode: str = "enhanced_grayscale",
) -> Dict[str, Any]:
    """
    Full pipeline to clean and prepare receipt images for OCR.
    
    Args:
        return_mode: 'enhanced_grayscale' (recommended for neural OCR like RapidOCR/PaddleOCR)
                     or 'binary' (traditional binarization, ideal for legacy Tesseract).
    
    Returns a dictionary with:
        - 'final': The final processed image ready for OCR
        - 'skew_angle': Detected skew in degrees
        - 'stages': Dictionary containing intermediate images for visual inspection:
            - 'original'
            - 'grayscale'
            - 'deskewed'
            - 'lighting_corrected'
            - 'denoised'
            - 'binarized'
    """
    original = load_image(image_input)
    stages: Dict[str, np.ndarray] = {"original": original}

    # Step 1: Grayscale
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original.copy()
    stages["grayscale"] = gray.copy()

    # Step 2: Deskew
    skew_angle = 0.0
    if enable_deskew:
        gray, skew_angle = deskew(gray)
        stages["deskewed"] = gray.copy()
    else:
        stages["deskewed"] = gray.copy()

    # Step 3: Lighting and shadow correction
    if enable_lighting_correction:
        corrected = correct_lighting(gray)
        stages["lighting_corrected"] = corrected.copy()
    else:
        corrected = gray.copy()
        stages["lighting_corrected"] = corrected.copy()

    # Step 4: Denoise
    if enable_denoise:
        denoised = denoise(corrected)
        stages["denoised"] = denoised.copy()
    else:
        denoised = corrected.copy()
        stages["denoised"] = denoised.copy()

    # Step 5: Binarization / Thresholding
    binary = binarize(denoised, method=binarize_method)
    stages["binarized"] = binary.copy()

    # Select output image based on OCR engine compatibility
    final_output = binary if return_mode == "binary" else denoised

    return {
        "final": final_output,
        "skew_angle": round(skew_angle, 2),
        "deskewed": stages["deskewed"],
        "illuminated": stages["lighting_corrected"],
        "binary": stages["binarized"],
        "stages": stages,
    }
