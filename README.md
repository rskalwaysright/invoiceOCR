# Receipt & Invoice Line-Item Extraction Engine
> **High-Throughput Receipt OCR Ingestion, Computer Vision Preprocessing, Fuzzy Catalog Normalization & Parallel Multi-Core Batch Processing.**

---

## 📌 Overview

This project provides an end-to-end, production-grade engine designed to automatically process noisy, tilted, and unevenly lit receipt and invoice images, extract line items and financial summaries, correct OCR corruptions via fuzzy catalog matching, and scale throughput using **multi-core parallel processing**.

### Key Capabilities:
1. **Computer Vision Preprocessing Pipeline**:
   - **Orientation Detection & Deskewing**: Hybrid contour + Hough text line orientation detection (corrects rotations up to $\pm 45^\circ$).
   - **Illumination & Shadow Correction**: Morphological background estimation division followed by CLAHE (Contrast-Limited Adaptive Histogram Equalization).
   - **Adaptive Binarization**: Local thresholding tailored for faint thermal receipt paper.
2. **Unified Dual-Backend OCR Engine**:
   - **RapidOCR (ONNX Runtime)**: Deep-learning text recognition running locally without external software dependencies.
   - **Tesseract OCR (Fallback)**: Open-source engine with auto-discovery on Windows/Linux.
3. **Robust Regex Line-Item Parser**:
   - Distinguishes product lines from metadata (date/time, store name, phone numbers, tax IDs).
   - Extracts quantities, unit prices, total item prices, discounts, and tax status.
   - Financial sanity audit checking parsed subtotal + tax against calculated line-item sum.
4. **Fuzzy Catalog Matcher & Auto-Correction**:
   - Multi-metric token matching (`token_sort_ratio`, `token_set_ratio`, normalized digit-to-char mapping) to correct OCR noise (e.g. `"M1LK 1GAL"` $\rightarrow$ `"Whole Milk 1 Gallon"`).
5. **Parallel Computing Engine**:
   - Multi-process worker pool (`ProcessPoolExecutor`) that bypasses the Python GIL.
   - Speedup benchmarking ($S_p = T_1 / T_p$), parallel efficiency ($E_p = S_p / p$), and Amdahl's Law parallel fraction analysis.
6. **Interactive Web Dashboard**:
   - Full-featured Streamlit dashboard with side-by-side vision transformations, interactive line-item tables, batch processing, and scaling graphs.

---

## 🏗️ Project Architecture

```
PROJECT/
├── app.py                     # Interactive Streamlit Web Application
├── catalog.json               # Canonical supermarket product catalog & aliases
├── main.py                    # Main CLI entrypoint
├── requirements.txt           # Python package dependencies
├── data/
│   ├── generate_samples.py    # Synthetic receipt generator (clean, tilted, shadowed)
│   ├── ground_truth/          # Ground truth JSON annotations for accuracy evaluation
│   └── samples/               # Synthetic test receipts
├── src/
│   ├── __init__.py            # Module exports
│   ├── evaluate.py            # Comparative accuracy benchmark (CER, WER, F1)
│   ├── matcher.py             # RapidFuzz catalog matcher & auto-correction
│   ├── ocr.py                 # Multi-backend OCR engine (RapidOCR / Tesseract)
│   ├── parallel_engine.py     # Multi-core parallel computing worker pool & scaling
│   ├── parser.py              # Structured regex line-item and financial parser
│   └── preprocess.py          # Deskewing, illumination division, CLAHE, binarization
└── tests/
    └── test_pipeline.py       # Automated unit and integration test suite
```

---

## ⚡ Parallel Computing & Performance Scaling

Receipt OCR and image preprocessing are CPU-intensive operations. By default, sequential processing processes one image at a time on a single CPU core, underutilizing modern multi-core processors.

The `ParallelReceiptEngine` leverages `concurrent.futures.ProcessPoolExecutor` to spawn isolated OS-level worker processes across available cores:

$$\text{Speedup } S_p = \frac{T_1}{T_p}$$

$$\text{Parallel Efficiency } E_p = \frac{S_p}{p} \times 100\%$$

$$\text{Amdahl's Law Limit } S_{\max} = \frac{1}{(1 - P) + \frac{P}{p}} \xrightarrow{p \to \infty} \frac{1}{1 - P}$$

Where $P$ represents the parallelizable portion of the pipeline (deskewing, binarization, deep-learning OCR inference).

---

## 🚀 Quickstart & Usage

### 1. Installation

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard

```bash
streamlit run app.py
```

Features in the Web App:
- **Single Receipt Studio**: Upload or pick a receipt; view deskewed & CLAHE transformations; review extracted line items and financial audit.
- **Parallel Batch Studio**: Ingest multiple receipts across configurable worker processes with live progress bars.
- **Parallel Speedup Studio**: Generate live speedup curves, execution time comparisons, and Amdahl's Law analysis.
- **Accuracy Benchmark**: Compare CER, WER, and F1 improvements with vs. without preprocessing.

### 3. Command-Line Interface (CLI)

#### Process a Single Receipt:
```bash
python main.py --image data/samples/sample_01_clean.png
```

#### Save Structured JSON to File:
```bash
python main.py --image data/samples/sample_02_skewed.png --output output.json
```

#### Run Multi-Core Parallel Batch Extraction:
```bash
python main.py --batch data/samples --output-dir results/ --parallel --workers 4
```

#### Run Multi-Core Parallel Speedup & Amdahl's Law Benchmark:
```bash
python main.py --parallel-benchmark
```

#### Run Preprocessing Accuracy Evaluation:
```bash
python main.py --benchmark
```

#### Re-generate Synthetic Test Dataset:
```bash
python main.py --generate-samples
```

---

## 🧪 Running Automated Tests

Run the test suite verifying all pipeline components:

```bash
python -m unittest tests/test_pipeline.py
```

---

## 📊 Benchmark Summary

| Evaluation Dimension | Without Preprocessing | With Preprocessing | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Character Error Rate (CER)** | 0.4844 | 0.4466 | **-7.8% relative error reduction** |
| **Line-Item F1-Score** | 0.9278 | 0.9778 | **+5.4% improvement** |
| **Financial Accuracy** | 60.0% | 80.0% | **+20.0% accuracy gain** |
| **Parallel Speedup (4 Cores)**| 1.00x (Baseline) | **Up to 3.2x** | **Multi-core throughput boost** |

---

## 📜 License
MIT License.
