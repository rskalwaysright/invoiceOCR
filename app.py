"""
Receipt & Invoice Line-Item Extraction Dashboard
Interactive Web Application with Parallel Computing Studio.
"""

import os
import sys
import time
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.preprocess import preprocess_receipt
from src.ocr import OCREngine
from src.parser import ReceiptParser
from src.matcher import CatalogMatcher
from src.evaluate import PipelineEvaluator
from src.parallel_engine import ParallelReceiptEngine

# Page configuration
st.set_page_config(
    page_title="Receipt & Invoice Extraction Engine",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Slate & Vibrant Accents)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ocr_engine(backend: str):
    return OCREngine(preferred_backend=backend)


@st.cache_resource
def load_catalog(path: str = "catalog.json"):
    return CatalogMatcher(path) if os.path.exists(path) else None


# Header
st.markdown('<div class="main-header">🧾 Receipt & Invoice Extraction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Deskewing, CLAHE Illumination Correction, OCR, Fuzzy Catalog Normalization & Parallel Multi-Core Ingestion</div>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    ocr_backend_choice = st.selectbox(
        "OCR Backend",
        ["auto", "rapidocr", "tesseract"],
        help="RapidOCR uses fast local ONNX runtime; Tesseract uses system Tesseract binary."
    )
    enable_prep = st.checkbox("Enable Image Preprocessing", value=True, help="Deskewing, shadow removal, and contrast enhancement.")
    enable_matching = st.checkbox("Enable Catalog Matching", value=True, help="Fuzzy lookup to correct OCR corruptions against catalog.json.")
    match_threshold = st.slider("Fuzzy Match Threshold", min_value=50.0, max_value=95.0, value=65.0, step=1.0)
    
    st.divider()
    st.subheader("🖥️ Hardware Concurrency")
    cpu_cores = os.cpu_count() or 4
    st.info(f"Detected **{cpu_cores} Logical CPU Cores** on host system.")


tabs = st.tabs([
    "🔍 Single Receipt Studio",
    "⚡ Parallel Batch Processing",
    "📈 Parallel Speedup & Scaling",
    "📊 Preprocessing & Accuracy Benchmark",
])

# -------------------------------------------------------------
# TAB 1: Single Receipt Studio
# -------------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Source")
        sample_options = [
            "Upload Custom Image...",
            "Sample 1: Clean High-Res",
            "Sample 2: Skewed (+5° Tilt)",
            "Sample 3: Shadow Gradient",
            "Sample 4: Skewed + Harsh Shadow",
            "Sample 5: Extreme Skew (-12° Tilt)",
        ]
        sample_map = {
            "Sample 1: Clean High-Res": "data/samples/sample_01_clean.png",
            "Sample 2: Skewed (+5° Tilt)": "data/samples/sample_02_skewed.png",
            "Sample 3: Shadow Gradient": "data/samples/sample_03_shadow.png",
            "Sample 4: Skewed + Harsh Shadow": "data/samples/sample_04_skewed_shadow.png",
            "Sample 5: Extreme Skew (-12° Tilt)": "data/samples/sample_05_extreme_skew.png",
        }

        selected_sample = st.selectbox("Choose Sample or Upload", sample_options)
        uploaded_file = None
        img_path = None

        if selected_sample == "Upload Custom Image...":
            uploaded_file = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.read())
                    img_path = tmp.name
        else:
            img_path = sample_map[selected_sample]

        if img_path and os.path.exists(img_path):
            input_pil = Image.open(img_path)
            st.image(input_pil, caption=f"Original Input: {os.path.basename(img_path)}", use_container_width=True)

    with col2:
        if img_path and os.path.exists(img_path):
            st.subheader("Pipeline Execution & Extraction")
            
            with st.spinner("Processing image through vision pipeline..."):
                t0 = time.perf_counter()
                
                # Step 1: Preprocess
                prep_result = None
                skew_angle = 0.0
                if enable_prep:
                    ocr_mode = "enhanced_grayscale" if ocr_backend_choice != "tesseract" else "binary"
                    prep_result = preprocess_receipt(img_path, return_mode=ocr_mode)
                    proc_img = prep_result["final"]
                    skew_angle = prep_result["skew_angle"]
                else:
                    proc_img = img_path

                # Step 2: OCR
                ocr = load_ocr_engine(ocr_backend_choice)
                extracted_text = ocr.extract_text(proc_img)

                # Step 3: Parsing
                parser = ReceiptParser()
                parsed = parser.parse(extracted_text)

                # Step 4: Catalog Matching
                matcher = load_catalog() if enable_matching else None
                if matcher:
                    parsed["line_items"] = matcher.match_all(parsed["line_items"], threshold=match_threshold)

                t_dur = time.perf_counter() - t0

            # Preprocessing Visual Comparison Expander
            if enable_prep and prep_result:
                with st.expander("🖼️ View Computer Vision Preprocessing Transformations", expanded=True):
                    vis_col1, vis_col2, vis_col3 = st.columns(3)
                    with vis_col1:
                        st.caption(f"Deskewed ({skew_angle:+.2f}° correction)")
                        st.image(prep_result["deskewed"], use_container_width=True)
                    with vis_col2:
                        st.caption("Illumination / CLAHE")
                        st.image(prep_result["illuminated"], use_container_width=True)
                    with vis_col3:
                        st.caption("Adaptive Binarization")
                        st.image(prep_result["binary"], use_container_width=True)

            # Metadata Metrics Bar
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Merchant", parsed.get("merchant_name") or "Unknown")
            m2.metric("Date & Time", f"{parsed.get('date') or '--'} {parsed.get('time') or ''}")
            m3.metric("Detected Tilt", f"{skew_angle:+.2f}°")
            m4.metric("Process Time", f"{t_dur:.2f}s")

            # Structured Line Items
            items = parsed.get("line_items", [])
            st.markdown(f"#### 🛒 Extracted Line Items ({len(items)})")
            
            if items:
                table_rows = []
                for it in items:
                    cat_match = it.get("catalog_match", {})
                    matched_bool = cat_match.get("matched", False)
                    corrected_bool = cat_match.get("corrected", False)

                    status = "✅ Matched" if matched_bool else "⚠️ Raw"
                    if corrected_bool:
                        status = "🔄 Corrected"

                    table_rows.append({
                        "Status": status,
                        "Item Name": it.get("matched_name") or it.get("description"),
                        "Raw OCR Text": it.get("description"),
                        "Qty": it.get("quantity", 1.0),
                        "Unit ($)": f"${it.get('unit_price', 0.0):.2f}",
                        "Total ($)": f"${it.get('total_price', 0.0):.2f}",
                        "SKU": it.get("sku") or "--",
                        "Category": it.get("category") or "Uncategorized",
                        "Match Score": f"{cat_match.get('confidence', 0)}%" if matched_bool else "--",
                    })

                df_items = pd.DataFrame(table_rows)
                st.dataframe(df_items, use_container_width=True, hide_index=True)
            else:
                st.warning("No line items detected. Check OCR output or preprocessing.")

            # Financial Sanity Audit
            fin = parsed.get("financial_summary", {})
            st.markdown("#### 💳 Financial Audit & Sanity Validation")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            fcol1.metric("Parsed Subtotal", f"${fin.get('parsed_subtotal') or 0.0:.2f}")
            fcol2.metric("Parsed Tax", f"${fin.get('parsed_tax') or 0.0:.2f}")
            fcol3.metric("Parsed Total", f"${fin.get('parsed_total') or 0.0:.2f}")
            
            calc_sum = fin.get("calculated_items_sum", 0.0)
            diff = fin.get("discrepancy", 0.0)
            
            if diff is not None and diff < 0.05:
                fcol4.success(f"Audit Passed (Diff: ${diff:.2f})")
            else:
                fcol4.warning(f"Audit Diff: ${diff if diff is not None else 0.0:.2f}")

            # Export Section
            st.download_button(
                label="📥 Download Structured JSON",
                data=json.dumps(parsed, indent=2),
                file_name="extracted_receipt.json",
                mime="application/json",
            )
        else:
            st.info("👈 Please select a sample receipt or upload an image to begin extraction.")

# -------------------------------------------------------------
# TAB 2: Parallel Batch Processing Studio
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("⚡ High-Throughput Parallel Batch Ingestion")
    st.markdown("Distribute receipt processing across multiple CPU cores using **ProcessPoolExecutor** to eliminate GIL bottlenecks.")

    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        batch_folder = st.text_input("Receipts Folder Path", value="data/samples")
        worker_slider = st.slider("Worker Processes", min_value=1, max_value=min(cpu_cores, 16), value=min(cpu_cores, 4))
        run_batch_btn = st.button("🚀 Run Parallel Batch Extraction", type="primary")

    with p_col2:
        if run_batch_btn:
            if not os.path.exists(batch_folder):
                st.error(f"Folder '{batch_folder}' does not exist.")
            else:
                image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                all_images = [
                    os.path.join(batch_folder, f)
                    for f in os.listdir(batch_folder)
                    if os.path.splitext(f)[1].lower() in image_exts
                ]

                if not all_images:
                    st.warning("No image files found in specified folder.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    engine = ParallelReceiptEngine(default_workers=worker_slider)
                    status_text.info(f"Processing {len(all_images)} receipts across {worker_slider} worker processes...")

                    def on_progress(done, total, res):
                        progress_bar.progress(done / total)
                        status_text.text(f"Processed {done}/{total} receipts ({(done/total)*100:.0f}%)")

                    batch_res = engine.process_batch_parallel(
                        image_paths=all_images,
                        num_workers=worker_slider,
                        enable_preprocess=enable_prep,
                        catalog_path="catalog.json" if enable_matching else "",
                        ocr_backend=ocr_backend_choice,
                        progress_callback=on_progress,
                    )

                    status_text.success("Batch processing complete!")
                    
                    # Performance Metrics
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Receipts Processed", batch_res["total_receipts"])
                    b2.metric("Wall-Clock Time", f"{batch_res['wall_clock_time_sec']:.2f}s")
                    b3.metric("Throughput", f"{batch_res['throughput_receipts_per_sec']:.2f} rps")
                    b4.metric("Avg Latency / Receipt", f"{batch_res['average_time_per_receipt_sec']:.2f}s")

                    # Results table
                    summary_rows = []
                    for itm in batch_res["results"]:
                        if itm.get("status") == "success":
                            res = itm["result"]
                            summary_rows.append({
                                "File": os.path.basename(itm["image_path"]),
                                "Worker PID": res["metadata"].get("worker_pid"),
                                "Store": res.get("merchant_name") or "--",
                                "Date": res.get("date") or "--",
                                "Items": len(res.get("line_items", [])),
                                "Total ($)": f"${res.get('financial_summary', {}).get('parsed_total') or 0.0:.2f}",
                                "Time (s)": f"{res['metadata'].get('processing_time_sec'):.2f}s",
                            })
                    if summary_rows:
                        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Parallel Speedup & Scaling Benchmark
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("📈 Multi-Core Parallel Scaling & Amdahl's Law Evaluation")
    st.markdown("""
    Evaluate system throughput, speedup factor ($S_p = T_1 / T_p$), and parallel efficiency ($E_p = S_p / p$) 
    across worker thread configurations.
    """)

    bench_run_btn = st.button("🔥 Run Multi-Core Benchmark", type="primary")

    if bench_run_btn:
        samples_dir = "data/samples"
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if os.path.splitext(f)[1].lower() in image_exts
        ]

        with st.spinner("Benchmarking worker configurations (1, 2, 4 cores)..."):
            engine = ParallelReceiptEngine()
            core_tests = [1, 2]
            if engine.cpu_cores >= 4:
                core_tests.append(4)
            if engine.cpu_cores >= 8:
                core_tests.append(8)

            bench_data = engine.benchmark_scaling(
                image_paths=images,
                worker_counts=core_tests,
                enable_preprocess=enable_prep,
                catalog_path="catalog.json",
                ocr_backend=ocr_backend_choice,
            )

        df_bench = pd.DataFrame(bench_data["scaling_results"])
        
        # Metric highlights
        c1, c2, c3 = st.columns(3)
        c1.metric("Sequential Baseline (1 Core)", f"{bench_data['baseline_sequential_time_sec']:.2f}s")
        max_speed = df_bench["speedup"].max()
        c2.metric("Max Speedup Achieved", f"{max_speed:.2f}x")
        c3.metric("Amdahl's Parallel Fraction (P)", f"{bench_data['estimated_parallel_fraction']*100:.1f}%")

        # Results table
        st.dataframe(
            df_bench.rename(columns={
                "workers": "Cores / Workers",
                "wall_time_sec": "Execution Time (s)",
                "speedup": "Speedup Factor (x)",
                "efficiency_pct": "Parallel Efficiency (%)",
                "throughput_rps": "Throughput (receipts/sec)",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Speedup Factor vs Worker Count")
            chart_df = df_bench.set_index("workers")[["speedup"]]
            chart_df["Linear (Ideal)"] = chart_df.index
            st.line_chart(chart_df)

        with chart_col2:
            st.markdown("#### Wall-Clock Execution Time (Seconds)")
            st.bar_chart(df_bench.set_index("workers")["wall_time_sec"])

        st.info(f"💡 **Theoretical Limit**: Based on Amdahl's Law, maximum attainable speedup with infinite cores is **{bench_data['theoretical_max_speedup_amdahl']:.2f}x** due to sequential I/O overhead.")

# -------------------------------------------------------------
# TAB 4: Accuracy Benchmark
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("📊 Computer Vision Preprocessing Accuracy Benchmark")
    st.markdown("Quantify the exact impact of deskewing, shadow division, and CLAHE normalization on OCR extraction accuracy.")

    run_acc_btn = st.button("🧪 Run Accuracy Evaluation", type="primary")

    if run_acc_btn:
        with st.spinner("Evaluating test receipts against ground truth..."):
            evaluator = PipelineEvaluator(catalog_path="catalog.json")
            eval_summary = evaluator.evaluate_dataset("data/samples", "data/ground_truth")

        if "error" in eval_summary:
            st.error(eval_summary["error"])
        else:
            w_out = eval_summary["without_preprocessing"]
            w_in = eval_summary["with_preprocessing"]
            delta = eval_summary["delta"]

            k1, k2, k3 = st.columns(3)
            k1.metric("CER Improvement", f"{-delta.get('cer_reduction', 0):.4f}", delta=f"{delta.get('cer_reduction', 0)*100:+.1f}% error reduction")
            k2.metric("Line-Item F1 Gain", f"{w_in.get('mean_line_item_f1', 0):.4f}", delta=f"{delta.get('f1_score_gain', 0):+.4f} F1 gain")
            k3.metric("Total Accuracy", f"{w_in.get('total_accuracy_pct', 0):.1f}%", delta=f"{delta.get('total_accuracy_gain_pct', 0):+.1f}%")

            comp_df = pd.DataFrame([
                {
                    "Metric": "Character Error Rate (CER)",
                    "Raw (Without Preprocessing)": f"{w_out.get('mean_cer', 0):.4f}",
                    "Preprocessed (With Preprocessing)": f"{w_in.get('mean_cer', 0):.4f}",
                    "Net Gain / Improvement": f"{delta.get('cer_reduction', 0):+.4f}",
                },
                {
                    "Metric": "Line-Item F1-Score",
                    "Raw (Without Preprocessing)": f"{w_out.get('mean_line_item_f1', 0):.4f}",
                    "Preprocessed (With Preprocessing)": f"{w_in.get('mean_line_item_f1', 0):.4f}",
                    "Net Gain / Improvement": f"{delta.get('f1_score_gain', 0):+.4f}",
                },
                {
                    "Metric": "Total Extraction Accuracy",
                    "Raw (Without Preprocessing)": f"{w_out.get('total_accuracy_pct', 0):.1f}%",
                    "Preprocessed (With Preprocessing)": f"{w_in.get('total_accuracy_pct', 0):.1f}%",
                    "Net Gain / Improvement": f"{delta.get('total_accuracy_gain_pct', 0):+.1f}%",
                },
            ])
            st.table(comp_df)
