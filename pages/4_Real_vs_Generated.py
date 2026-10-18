"""
Streamlit Page 4: Real vs. Generated Comparison
Side-by-side qualitative and distributional visual comparisons.
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
from utils.checkpoint import get_latest_checkpoint, load_dbn_checkpoint
from utils.data_loader import get_sample_real_images
from utils.visualization import plot_real_vs_generated, plot_pixel_distribution, plot_digit_grid

st.set_page_config(page_title="Real vs Generated | DBN", page_icon="⚖️", layout="wide")

paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config()

st.title("⚖️ Real vs. Generated Handwritten Digit Comparison")
st.markdown("Direct qualitative and statistical comparison between real human-written MNIST digits and DBN-synthesized digits.")
st.markdown("---")

latest_ckpt = get_latest_checkpoint(paths.CHECKPOINT_DIR)

if latest_ckpt is None:
    st.warning("⚠️ No trained checkpoint found. Please train the DBN on **Page 2** first.")
else:
    c_opts1, c_opts2, c_opts3 = st.columns(3)
    with c_opts1:
        num_pairs = st.slider("Number of comparison pairs:", min_value=5, max_value=20, value=10, step=1)
    with c_opts2:
        gibbs_steps = st.slider("Gibbs Sampling Steps for Gen:", min_value=50, max_value=300, value=100, step=25)
    with c_opts3:
        seed = st.number_input("Comparison Seed:", min_value=1, max_value=9999, value=42, step=1)

    run_comp = st.button("🔄 Generate & Compare", type="primary", use_container_width=True)

    # Load or generate comparison batch
    if run_comp or "comp_real" not in st.session_state:
        with st.spinner("Fetching real digits and generating synthetic samples..."):
            try:
                real_imgs, _ = get_sample_real_images(paths.DATA_DIR, num_samples=num_pairs, seed=seed)
                model, _ = load_dbn_checkpoint(latest_ckpt, device=torch.device("cpu"))
                gen_imgs = model.generate(num_samples=num_pairs, gibbs_steps=gibbs_steps, random_seed=seed)
                st.session_state["comp_real"] = real_imgs
                st.session_state["comp_gen"] = gen_imgs
            except Exception as e:
                st.error(f"Failed to prepare comparison: {e}")

    if "comp_real" in st.session_state and "comp_gen" in st.session_state:
        real_imgs = st.session_state["comp_real"]
        gen_imgs = st.session_state["comp_gen"]

        st.subheader("1. Side-by-Side Visual Comparison")
        st.markdown("**Top Row:** Real Human MNIST Digits | **Bottom Row:** DBN-Synthesized Digits")
        fig_comp = plot_real_vs_generated(real_imgs, gen_imgs, num_samples=num_pairs)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        st.markdown("---")
        st.subheader("2. Grids Comparison")
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.markdown("##### 📌 Real MNIST Digits")
            fig_r = plot_digit_grid(real_imgs, cols=min(5, num_pairs))
            st.pyplot(fig_r)
            plt.close(fig_r)

        with g_c2:
            st.markdown("##### 🤖 DBN Synthetic Digits")
            fig_g = plot_digit_grid(gen_imgs, cols=min(5, num_pairs))
            st.pyplot(fig_g)
            plt.close(fig_g)

        st.markdown("---")
        st.subheader("3. Pixel Intensity Distribution Histogram")
        st.markdown("Compares whether the synthetic model preserves the true bimodal density of digit strokes vs background.")
        fig_dist = plot_pixel_distribution(real_imgs, gen_imgs)
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        st.markdown("---")
        st.subheader("4. Critical Academic Comparison & Discussion")
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown("""
            #### Similarities:
            - **Global Topologies:** Both real and synthetic digits display characteristic handwritten loops (0, 6, 8), vertical strokes (1), and crossbars (4, 7).
            - **Background Sparsity:** Both datasets have strong sparsity (>75% of pixels close to 0.0), showing the DBN learned the inactive canvas regions.
            - **Centering:** Generated digits naturally emerge centered in the 28x28 bounding box, matching MNIST training distribution.
            """)
        with col_diag2:
            st.markdown("""
            #### Differences & DBN Limitations:
            - **Fuzzy Margins:** Real digits have crisp pen strokes; DBN digits show diffuse, probabilistic halos around stroke perimeters due to Bernoulli expectation.
            - **Stroke Artifacts:** Occasional faint background speckles appear where hidden units have nonzero conditional activation.
            - **Ambiguous Digits:** In undirected energy models, saddle points between digit classes can produce hybrid numerals (e.g. blend of a 3 and a 5).
            """)
