"""
Streamlit Page 2: Model Training
Interactive Greedy Layer-Wise Pretraining of Deep Belief Network with Live Diagnostics.
"""

import sys
import time
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import get_config, DEMO_MODE_CONFIG, FULL_MODE_CONFIG
from models.train import train_dbn_pipeline
from utils.checkpoint import get_latest_checkpoint, load_dbn_checkpoint
from utils.visualization import plot_training_curves

st.set_page_config(page_title="Model Training | DBN", page_icon="⚙️", layout="wide")

paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config()

st.title("⚙️ Layer-Wise Pretraining of Deep Belief Network")
st.markdown("Train stacked Restricted Boltzmann Machines (RBMs) sequentially using **Contrastive Divergence (CD-1)**.")
st.markdown("---")

# Layout: Configuration on Left, Progress & Existing Checkpoints on Right
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Training Configuration & Presets")

    mode_choice = st.radio(
        "Select Training Preset:",
        options=["Quick Demo Mode (Laptop CPU Friendly)", "Full Training Mode (Deep Multi-layer)"],
        index=0,
        help="Demo mode uses smaller layer dimensions and a 5,000-sample subset to complete rapidly on CPU."
    )

    is_demo = "Demo" in mode_choice
    selected_mode = "demo" if is_demo else "full"
    preset = DEMO_MODE_CONFIG if is_demo else FULL_MODE_CONFIG

    st.info(f"""
    **Selected Preset Details:**
    - **Architecture:** `{preset['layer_dims']}` (3 Stacked RBMs)
    - **Training Subset:** `{preset['max_train_samples']:,}` samples
    - **Default Epochs:** `{preset['epochs_per_rbm']}` per RBM
    - **CD Sampling Steps (k):** `{preset['k_cd']}`
    """)

    with st.expander("🛠️ Advanced Hyperparameter Overrides", expanded=False):
        epochs_override = st.slider("Epochs per RBM layer:", min_value=1, max_value=20, value=preset['epochs_per_rbm'], step=1)
        batch_size_override = st.selectbox("Batch Size:", options=[32, 64, 128], index=1)
        lr_override = st.slider("Learning Rate:", min_value=0.01, max_value=0.15, value=float(preset['learning_rate']), step=0.01)

    train_btn = st.button("🚀 Start DBN Pretraining", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Checkpoint Status & Diagnostics")
    latest_ckpt = get_latest_checkpoint(paths.CHECKPOINT_DIR)

    if latest_ckpt is not None:
        st.success(f"**Loaded Checkpoint:** `{latest_ckpt.name}`")
        try:
            model, meta = load_dbn_checkpoint(latest_ckpt, device=torch.device("cpu"))
            st.write(f"- **Layers:** `{model.layer_dims}`")
            st.write(f"- **Pretraining Mode:** `{meta.get('training_mode', 'N/A').upper()}`")
            st.write(f"- **Created At:** `{meta.get('timestamp', 'N/A')}`")

            hist = meta.get("history", {})
            if hist:
                st.write("##### Pretraining Loss & Reconstruction Curves")
                fig = plot_training_curves(hist)
                st.pyplot(fig)
                plt.close(fig)
        except Exception as e:
            st.warning(f"Could not load metadata from checkpoint: {e}")
    else:
        st.info("No saved model checkpoints yet. Configure settings on the left and click 'Start DBN Pretraining'.")

# ----------------- LIVE TRAINING EXECUTION -----------------
if train_btn:
    st.markdown("---")
    st.subheader("🏋️ Live Training Execution")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    metric_cols = st.columns(3)
    m1 = metric_cols[0].empty()
    m2 = metric_cols[1].empty()
    m3 = metric_cols[2].empty()

    total_layers = len(preset['layer_dims']) - 1
    total_epochs_all = total_layers * epochs_override

    def on_progress(rbm_idx, epoch, total_epochs, loss, recon, msg):
        overall_step = (rbm_idx * total_epochs) + epoch
        fraction = min(1.0, overall_step / total_epochs_all)
        progress_bar.progress(fraction)
        status_text.markdown(f"**Active Stage:** Training RBM {rbm_idx + 1}/{total_layers} | Epoch {epoch}/{total_epochs}")
        m1.metric("Current RBM", f"RBM {rbm_idx + 1}")
        m2.metric("CD Surrogate Loss", f"{loss:.4f}")
        m3.metric("Reconstruction MSE", f"{recon:.4f}")

    with st.spinner("Training Deep Belief Network layer by layer..."):
        try:
            results = train_dbn_pipeline(
                mode=selected_mode,
                epochs_override=epochs_override,
                batch_size_override=batch_size_override,
                learning_rate_override=lr_override,
                progress_callback=on_progress
            )
            progress_bar.progress(1.0)
            status_text.success("🎉 Pretraining Pipeline Successfully Completed!")
            st.balloons()

            st.success(f"Model saved to: `{results['checkpoint_path']}`")
            st.write(f"Total Pretraining Time: **{results['elapsed_seconds']:.2f} seconds**")

            st.subheader("Diagnostic Curves from Completed Run")
            fig = plot_training_curves(results["history"])
            st.pyplot(fig)
            plt.close(fig)

        except Exception as err:
            st.error(f"Training failed with error: {err}")
            st.exception(err)
