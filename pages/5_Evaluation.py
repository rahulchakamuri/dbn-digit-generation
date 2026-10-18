"""
Streamlit Page 5: Quantitative and Qualitative Evaluation
Reconstruction Error, Recognizability Evaluation, and Statistical Diagnostics.
"""

import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import get_config
from utils.checkpoint import get_latest_checkpoint, load_dbn_checkpoint
from utils.data_loader import get_sample_real_images
from models.evaluate import evaluate_dbn
from models.classifier import load_classifier, evaluate_recognizability

st.set_page_config(page_title="Evaluation & Metrics | DBN", page_icon="📈", layout="wide")

paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config()

st.title("📈 Quantitative & Qualitative Evaluation Suite")
st.markdown("Measure end-to-end representation learning, statistical alignment, and recognizability.")
st.markdown("---")

latest_ckpt = get_latest_checkpoint(paths.CHECKPOINT_DIR)

if latest_ckpt is None:
    st.warning("⚠️ No trained checkpoint found. Train the model on **Page 2** before running evaluation.")
else:
    c_e1, c_e2 = st.columns([1, 2])

    with c_e1:
        st.subheader("⚙️ Evaluation Parameters")
        st.write(f"- **Active Model:** `{latest_ckpt.name}`")
        eval_samples = st.selectbox("Sample count for evaluation:", [25, 50, 100], index=0)
        eval_gibbs = st.slider("Gibbs sampling steps:", min_value=50, max_value=300, value=100, step=25)
        eval_seed = st.number_input("Evaluation Seed:", min_value=1, max_value=9999, value=42, step=1)

        eval_btn = st.button("📊 Run Full Evaluation", type="primary", use_container_width=True)

    with c_e2:
        st.subheader("💡 Key Evaluation Principles")
        st.info("""
        **Academic Ground Rules for DBN Evaluation:**
        1. **Reconstruction MSE** measures how accurately the stacked RBMs preserve information in a round-trip pass (visible -> hidden -> visible).
        2. **Recognizability** uses a separately trained, independent MNIST CNN as an external judge to measure whether synthetic strokes trigger legitimate digit class detectors.
        3. **Important Distinction:** High classifier confidence does *not* prove human-level realism; it merely proves that generated stroke topologies align with standard digit features.
        """)

    if eval_btn or "eval_report" not in st.session_state:
        with st.spinner("Executing evaluation pipeline (Reconstruction, Sampling, Classifier Scoring)..."):
            try:
                report = evaluate_dbn(
                    checkpoint_path=latest_ckpt,
                    num_samples=eval_samples,
                    gibbs_steps=eval_gibbs,
                    seed=eval_seed
                )
                st.session_state["eval_report"] = report
            except Exception as err:
                st.error(f"Evaluation error: {err}")

    if "eval_report" in st.session_state:
        rep = st.session_state["eval_report"]

        st.markdown("---")
        st.subheader("1. Core Performance Metrics")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Reconstruction MSE", f"{rep['reconstruction_mse']:.4f}", delta="Lower is better", delta_color="inverse")
        with m_col2:
            st.metric("Real Background %", f"{rep['pixel_statistics']['real_background_sparsity_pct']:.1f}%")
        with m_col3:
            st.metric("Gen Background %", f"{rep['pixel_statistics']['gen_background_sparsity_pct']:.1f}%")
        with m_col4:
            if rep["classifier_evaluation"]:
                st.metric("Recognizability Ratio", f"{rep['classifier_evaluation']['recognizable_ratio'] * 100:.1f}%", help="Fraction of generated digits with >= 70% CNN classifier confidence")
            else:
                st.metric("Recognizability", "N/A (Classifier not trained)")

        st.markdown("---")
        st.subheader("2. Independent CNN Classifier Recognizability Assessment")

        if rep["classifier_evaluation"]:
            c_info = rep["classifier_evaluation"]
            c_left, c_right = st.columns(2)

            with c_left:
                st.markdown(f"""
                - **Confident Samples (Confidence $\\ge 70\\%$):** `{c_info['confident_count']} / {c_info['total_samples']}`
                - **Average Classifier Confidence:** `{c_info['average_confidence'] * 100:.2f}%`
                - **Confidence Threshold:** `{c_info['confidence_threshold'] * 100:.0f}%`
                """)
                st.warning(c_info["disclaimer"])

            with c_right:
                st.markdown("##### Predicted Digit Distribution in Synthetic Batch")
                dist_df = pd.DataFrame(
                    list(c_info["digit_distribution"].items()),
                    columns=["Digit Class", "Count"]
                )
                st.bar_chart(dist_df.set_index("Digit Class"))
        else:
            st.info("The external evaluation classifier was not found. It will be trained automatically during your next training run on Page 2.")

        st.markdown("---")
        st.subheader("3. Interactive Round-Trip Digit Reconstruction Inspector")
        st.markdown("Inspect how the trained DBN compresses and decompresses individual real MNIST digits:")

        recon_idx = st.slider("Select sample index:", 0, len(rep["real_images"]) - 1, 0)
        sample_real = rep["real_images"][recon_idx]

        model, _ = load_dbn_checkpoint(latest_ckpt, device=torch.device("cpu"))
        with torch.no_grad():
            sample_recon = model.reconstruct(sample_real.view(1, 784)).view(28, 28)

        r_col1, r_col2, r_col3 = st.columns([1, 1, 2])
        with r_col1:
            st.markdown("**Original Real Digit**")
            fig1, ax1 = plt.subplots(figsize=(2.5, 2.5))
            ax1.imshow(sample_real.numpy(), cmap="gray", vmin=0, vmax=1)
            ax1.axis("off")
            st.pyplot(fig1)
            plt.close(fig1)

        with r_col2:
            st.markdown("**DBN Reconstruction**")
            fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
            ax2.imshow(sample_recon.numpy(), cmap="gray", vmin=0, vmax=1)
            ax2.axis("off")
            st.pyplot(fig2)
            plt.close(fig2)

        with r_col3:
            st.markdown("**Per-Pixel Difference (Absolute Error)**")
            diff = torch.abs(sample_real - sample_recon).numpy()
            fig3, ax3 = plt.subplots(figsize=(2.5, 2.5))
            im3 = ax3.imshow(diff, cmap="hot", vmin=0, vmax=1)
            ax3.axis("off")
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            st.pyplot(fig3)
            plt.close(fig3)
