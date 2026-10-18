"""
Streamlit Page 1: Dataset Explorer
Explores the MNIST handwritten digit dataset, image representations, and statistics.
"""

import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import get_config
from utils.data_loader import load_mnist, get_sample_real_images, get_dataset_info
from utils.visualization import plot_digit_grid

st.set_page_config(page_title="Dataset Explorer | DBN", page_icon="📊", layout="wide")

paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config()

st.title("📊 MNIST Dataset Explorer")
st.markdown("Inspect real handwritten digit samples, dimensions, and pixel value distributions.")
st.markdown("---")

info = get_dataset_info(paths.DATA_DIR)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Training Set", f"{info['train_count']:,} images")
with col2:
    st.metric("Total Test Set", f"{info['test_count']:,} images")
with col3:
    st.metric("Resolution", info["dimensions"])
with col4:
    st.metric("Flattened Dimension", f"{info['features']} features")

st.markdown("### 🖼️ Real Digit Samples from MNIST")

c1, c2 = st.columns([1, 3])
with c1:
    num_samples = st.selectbox("Number of samples to preview:", [16, 25, 36, 49], index=1)
    seed = st.number_input("Random Seed:", min_value=1, max_value=9999, value=42, step=1)
    load_btn = st.button("🔄 Refresh Samples", use_container_width=True)

with c2:
    try:
        real_images, labels = get_sample_real_images(paths.DATA_DIR, num_samples=num_samples, seed=seed)
        titles = [f"Digit: {lbl.item()}" for lbl in labels]
        fig = plot_digit_grid(real_images, titles=titles)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Could not load samples: {e}")
        st.info("The MNIST dataset may still be downloading. Please retry shortly.")

st.markdown("---")
st.markdown("### 🔍 Pixel Value Distribution & Preprocessing Analysis")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Grayscale Pixel Characteristics")
    if 'real_images' in locals():
        np_imgs = real_images.numpy()
        mean_val = float(np_imgs.mean())
        std_val = float(np_imgs.std())
        sparsity = float((np_imgs < 0.1).mean() * 100)

        st.write(f"- **Pixel Value Range:** `[{np_imgs.min():.2f}, {np_imgs.max():.2f}]`")
        st.write(f"- **Mean Pixel Intensity:** `{mean_val:.4f}`")
        st.write(f"- **Standard Deviation:** `{std_val:.4f}`")
        st.write(f"- **Background Sparsity (< 0.1 intensity):** `{sparsity:.1f}%`")
        
        st.info("""
        **Why Bernoulli RBMs fit MNIST:**
        MNIST pixels are largely binary: ~80% of pixels are completely black background ($0.0$) and stroke centers are bright ($1.0$).
        Interpreting continuous pixel intensities in $[0, 1]$ as Bernoulli probabilities $P(v_i = 1)$ allows standard binary RBMs to reconstruct grayscale strokes cleanly without Gaussian noise complications.
        """)

with col_b:
    st.subheader("Pixel Intensity Histogram")
    if 'real_images' in locals():
        fig_hist, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np_imgs.flatten(), bins=40, color="#1f77b4", alpha=0.7, edgecolor="black")
        ax.set_title("Histogram of Real MNIST Pixels", fontsize=11)
        ax.set_xlabel("Pixel Value (0=Black, 1=White)")
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_hist)
        plt.close(fig_hist)
