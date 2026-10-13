"""
Sample Receipt Generator for Testing and Benchmarking.

Creates realistic test receipts:
1. Clean receipt
2. Skewed receipt (tilted angle)
3. Shadowed receipt (uneven lighting gradient)
4. Skewed + shadowed receipt (hard case)
5. Low-contrast receipt (fading thermal print)

Also outputs ground-truth JSON annotations to data/ground_truth/ for accuracy comparison.
"""

import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def create_receipt_text_and_gt(receipt_id: int):
    """Generate structured receipt text and corresponding ground truth metadata."""
    stores = [
        {"name": "METRO FRESH MARKET", "address": "1042 MARKET ST, SAN FRANCISCO, CA", "phone": "(415) 555-0199"},
        {"name": "VALLEY ORGANIC GROCERY", "address": "782 HIGHWAY 101, REDWOOD CITY, CA", "phone": "(650) 555-0144"},
        {"name": "CORNERSTONE SUPERMARKET", "address": "320 BROADWAY, NEW YORK, NY", "phone": "(212) 555-0182"},
    ]
    store = stores[(receipt_id - 1) % len(stores)]

    available_items = [
        {"desc": "WHOLE MILK 1GAL", "qty": 1.0, "unit": 4.29, "total": 4.29, "sku": "DAIRY-001"},
        {"desc": "GRADE A LARGE EGGS 12CT", "qty": 1.0, "unit": 3.49, "total": 3.49, "sku": "DAIRY-002"},
        {"desc": "SALTED BUTTER 16OZ", "qty": 1.0, "unit": 4.99, "total": 4.99, "sku": "DAIRY-003"},
        {"desc": "SOURDOUGH BREAD LOAF", "qty": 1.0, "unit": 4.49, "total": 4.49, "sku": "BAKERY-001"},
        {"desc": "ORGANIC BANANAS", "qty": 1.0, "unit": 1.99, "total": 1.99, "sku": "PRODUCE-001"},
        {"desc": "HASS AVOCADO", "qty": 2.0, "unit": 1.50, "total": 3.00, "sku": "PRODUCE-003"},
        {"desc": "PURE ORANGE JUICE 52OZ", "qty": 1.0, "unit": 4.19, "total": 4.19, "sku": "BEV-001"},
        {"desc": "GROUND COFFEE DARK 12OZ", "qty": 1.0, "unit": 8.99, "total": 8.99, "sku": "BEV-003"},
        {"desc": "JASMINE RICE 5LB", "qty": 1.0, "unit": 7.99, "total": 7.99, "sku": "PANTRY-002"},
        {"desc": "CHICKEN BREAST 2LB", "qty": 1.0, "unit": 8.99, "total": 8.99, "sku": "MEAT-001"},
        {"desc": "SEA SALT POTATO CHIPS 8OZ", "qty": 1.0, "unit": 3.99, "total": 3.99, "sku": "SNACK-001"},
        {"desc": "PAPER TOWELS 6DBL", "qty": 1.0, "unit": 10.99, "total": 10.99, "sku": "HOUSE-001"},
    ]

    # Pick 4-6 items deterministically for repeatability
    random.seed(receipt_id * 42)
    selected_items = random.sample(available_items, k=5)

    subtotal = round(sum(it["total"] for it in selected_items), 2)
    tax = round(subtotal * 0.0825, 2)
    total = round(subtotal + tax, 2)

    date_str = f"10/{10 + receipt_id:02d}/2026"
    time_str = f"{11 + receipt_id:02d}:35 PM"

    lines = [
        store["name"],
        store["address"],
        store["phone"],
        f"DATE: {date_str}   TIME: {time_str}",
        "----------------------------------------",
    ]

    gt_line_items = []
    for item in selected_items:
        if item["qty"] > 1.0:
            lines.append(f"{int(item['qty'])} @ {item['unit']:.2f}")
            lines.append(f"{item['desc']:<28} ${item['total']:>6.2f}")
        else:
            lines.append(f"{item['desc']:<28} ${item['total']:>6.2f}")

        gt_line_items.append({
            "description": item["desc"],
            "quantity": item["qty"],
            "unit_price": item["unit"],
            "total_price": item["total"],
            "sku": item["sku"],
        })

    lines.extend([
        "----------------------------------------",
        f"SUBTOTAL                     ${subtotal:>6.2f}",
        f"TAX (8.25%)                  ${tax:>6.2f}",
        f"TOTAL                        ${total:>6.2f}",
        "----------------------------------------",
        "PAYMENT METHOD: VISA **** 4812",
        "THANK YOU FOR SHOPPING WITH US!",
    ])

    full_text = "\n".join(lines)

    ground_truth = {
        "receipt_id": receipt_id,
        "merchant_name": store["name"],
        "date": date_str,
        "time": time_str,
        "full_text": full_text,
        "line_items": gt_line_items,
        "financial_summary": {
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
        },
    }

    return lines, ground_truth


def render_receipt_image(lines, width=650, line_height=28) -> np.ndarray:
    """Render receipt text lines into a realistic image with slight paper texture."""
    height = max(550, len(lines) * line_height + 80)

    # Base paper background (off-white 248, 246, 242)
    base = np.full((height, width, 3), (245, 245, 243), dtype=np.uint8)

    # Add subtle paper grain noise
    noise = np.random.normal(0, 3, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Convert to PIL for sharp text rendering
    img_pil = Image.fromarray(base)
    draw = ImageDraw.Draw(img_pil)

    # Use default bitmap font
    font = ImageFont.load_default()

    y = 35
    for line in lines:
        draw.text((35, y), line, fill=(35, 35, 38), font=font)
        y += line_height

    return np.array(img_pil)


def add_skew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle to simulate handheld camera tilt."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    # Fill background with dark table surface
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(45, 45, 50)
    )


def add_shadow(image: np.ndarray, intensity: float = 0.55) -> np.ndarray:
    """Add realistic diagonal shadow gradient simulating handheld camera shadow."""
    h, w = image.shape[:2]
    # Create diagonal gradient
    y_grad = np.linspace(1.0 - intensity, 1.0, h)[:, None]
    x_grad = np.linspace(1.0 - intensity, 1.0, w)[None, :]
    grad = (y_grad + x_grad) / 2.0

    shadowed = (image.astype(np.float32) * grad[:, :, None]).clip(0, 255).astype(np.uint8)
    return shadowed


def generate_all_samples(output_dir: str = "data"):
    """Generate sample images and ground truth JSONs."""
    samples_dir = os.path.join(output_dir, "samples")
    gt_dir = os.path.join(output_dir, "ground_truth")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    test_cases = [
        {"id": 1, "filename": "sample_01_clean.png", "skew": 0.0, "shadow": False},
        {"id": 2, "filename": "sample_02_skewed.png", "skew": 8.5, "shadow": False},
        {"id": 3, "filename": "sample_03_shadow.png", "skew": 0.0, "shadow": True},
        {"id": 4, "filename": "sample_04_skewed_shadow.png", "skew": -11.0, "shadow": True},
        {"id": 5, "filename": "sample_05_extreme_skew.png", "skew": 15.0, "shadow": False},
    ]

    for tc in test_cases:
        lines, gt = create_receipt_text_and_gt(tc["id"])
        img = render_receipt_image(lines)

        if tc["shadow"]:
            img = add_shadow(img, intensity=0.50)

        if abs(tc["skew"]) > 0.01:
            img = add_skew(img, tc["skew"])

        img_path = os.path.join(samples_dir, tc["filename"])
        cv2.imwrite(img_path, img)

        gt_filename = os.path.splitext(tc["filename"])[0] + ".json"
        gt_path = os.path.join(gt_dir, gt_filename)
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2)

        print(f"Generated {tc['filename']} and {gt_filename}")

    print(f"\nAll sample receipts generated in {samples_dir} with ground truth in {gt_dir}")


if __name__ == "__main__":
    generate_all_samples()
