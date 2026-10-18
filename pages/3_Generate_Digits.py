"""
Streamlit Page 3: Generate Synthetic Handwritten Digits
Sample novel digits using Top-RBM Gibbs Sampling and Downward Ancestral Propagation.
"""

import io
import sys
import zipfile
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import get_config
from models.generate import generate_digits
from utils.checkpoint import get_latest_checkpoint, load_dbn_checkpoint
from utils.visualization import plot_digit_grid, plot_gibbs_trajectory

st.set_page_config(page_title="Generate Digits | DBN", page_icon="🎨", layout="wide")

paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config()

st.title("🎨 Probabilistic Synthetic Digit Generation")
st.markdown("Synthesize new handwritten digits from the learned probability distribution of the Deep Belief Network.")
st.markdown("---")

latest_ckpt = get_latest_checkpoint(paths.CHECKPOINT_DIR)

if latest_ckpt is None:
    st.warning("⚠️ No trained model checkpoint detected. Please train a model first on **Page 2 (Model Training)**.")
else:
    col_ctrl, col_display = st.columns([1, 2])

    with col_ctrl:
        st.subheader("1. Sampling Parameters")
        st.info(f"**Loaded Checkpoint:** `{latest_ckpt.name}`")

        num_samples = st.selectbox(
            "Number of digits to generate:",
            options=[10, 25, 50, 100],
            index=1,
            help="Higher counts provide a better view of distribution diversity."
        )

        gibbs_steps = st.slider(
            "Top-RBM Gibbs Sampling Steps (M):",
            min_value=20,
            max_value=500,
            value=100,
            step=20,
            help="Number of alternating Gibbs steps in top associative memory. More steps allow better Markov chain equilibrium mixing."
        )

        random_seed = st.number_input(
            "Random Seed:",
            min_value=1,
            max_value=999999,
            value=42,
            step=1,
            help="Set seed for reproducible synthetic outputs."
        )

        st.markdown("""
        **Sampling Strategy Mechanics:**
        1. Initialize top layer $\\mathbf{h}_3$ with Bernoulli noise.
        2. Run $M$ steps of alternating Gibbs updates in RBM 3: $\\mathbf{h}_2 \\leftrightarrow \\mathbf{h}_3$.
        3. Project equilibrium state $\\mathbf{h}_2$ downwards through directed layers: $\\mathbf{h}_2 \\to \\mathbf{h}_1 \\to \\mathbf{v}$.
        """)

        gen_btn = st.button("✨ Generate Synthetic Digits", type="primary", use_container_width=True)

    with col_display:
        st.subheader("2. Synthetic Digits Preview")

        # Session state for generated images
        if "latest_generated_tensor" not in st.session_state:
            st.session_state["latest_generated_tensor"] = None

        if gen_btn:
            with st.spinner(f"Running {gibbs_steps} Gibbs sampling iterations and downward synthesis..."):
                try:
                    gen_tensor, saved_pngs, grid_path = generate_digits(
                        checkpoint_path=latest_ckpt,
                        num_samples=num_samples,
                        gibbs_steps=gibbs_steps,
                        random_seed=random_seed
                    )
                    st.session_state["latest_generated_tensor"] = gen_tensor
                    st.success(f"Generated {num_samples} synthetic digits successfully!")
                except Exception as err:
                    st.error(f"Generation failed: {err}")

        # Display current generated tensor if available
        if st.session_state["latest_generated_tensor"] is not None:
            imgs = st.session_state["latest_generated_tensor"]
            fig = plot_digit_grid(imgs, cols=min(10, int(len(imgs)**0.5) + 2))
            st.pyplot(fig)

            # Download Grid PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            c_down1, c_down2 = st.columns(2)
            with c_down1:
                st.download_button(
                    label="📥 Download Grid Collage (PNG)",
                    data=buf,
                    file_name=f"dbn_synthetic_{len(imgs)}_digits.png",
                    mime="image/png",
                    use_container_width=True
                )

            with c_down2:
                # Create in-memory zip of individual images
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    imgs_np = (imgs.numpy() * 255.0).astype("uint8")
                    for idx, img_arr in enumerate(imgs_np):
                        pil_img = Image.fromarray(img_arr, mode="L")
                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format="PNG")
                        zip_file.writestr(f"synthetic_digit_{idx+1:03d}.png", img_byte_arr.getvalue())
                zip_buffer.seek(0)

                st.download_button(
                    label="📦 Download All Digits (ZIP)",
                    data=zip_buffer,
                    file_name="synthetic_digits_pngs.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        else:
            st.info("Click **'✨ Generate Synthetic Digits'** to produce samples.")

    st.markdown("---")
    st.subheader("🌀 3. MCMC Gibbs Convergence Strip: Noise-to-Digit Evolution")
    st.markdown("""
    Observe how the Markov chain in the top associative memory navigates from unorganized random Bernoulli noise
    at Step 0 into structured energy minima representing coherent digit strokes by Step 100.
    """)

    c_tr1, c_tr2 = st.columns([1, 2])
    with c_tr1:
        traj_chains = st.slider("Parallel Markov Chains to trace:", min_value=3, max_value=6, value=4, step=1)
        traj_seed = st.number_input("MCMC Seed:", min_value=1, max_value=9999, value=42, step=1)
        run_traj_btn = st.button("🔬 Trace MCMC Trajectory", use_container_width=True)

    with c_tr2:
        if run_traj_btn or "mcmc_trajectory_fig" not in st.session_state:
            with st.spinner("Recording Gibbs sampling milestone reconstructions..."):
                try:
                    model, _ = load_dbn_checkpoint(latest_ckpt, device=torch.device("cpu"))
                    traj_dict = model.generate_trajectory(
                        num_samples=traj_chains,
                        milestones=[0, 5, 15, 30, 60, 100],
                        random_seed=traj_seed
                    )
                    fig_traj = plot_gibbs_trajectory(traj_dict)
                    st.session_state["mcmc_trajectory_fig"] = fig_traj
                except Exception as err:
                    st.error(f"Failed to generate trajectory: {err}")

        if "mcmc_trajectory_fig" in st.session_state:
            st.pyplot(st.session_state["mcmc_trajectory_fig"])

    st.markdown("---")
    st.subheader("💡 Academic Understanding: Visual Characteristics of DBN Samples")
    st.markdown("""
    When inspecting the generated images, you will observe several key characteristics inherent to Deep Belief Networks:
    1. **Structural Legibility:** Many generated samples form recognizable digit strokes (e.g. 0s, 1s, 4s, 7s, 8s).
    2. **Fuzzy Edges & Grayscale Halos:** Unlike modern GANs with adversarial discriminators that penalize blurriness, DBNs reconstruct visible probabilities $P(v_i = 1 | h)$. Intermediate pixels reflect stroke uncertainty.
    3. **Mode Mixing / Chimeric Digits:** In Markov chain Monte Carlo, the chain transitions between multiple energy minima. If sampling occurs during a transition between modes (e.g. between a 3 and an 8), a hybrid or ambiguous digit is generated. This is normal probabilistic behavior in energy models.
    """)

