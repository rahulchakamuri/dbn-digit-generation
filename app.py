"""
Deep Belief Network for Synthetic Handwritten Digit Generation
Main Entry Point and Dashboard (Streamlit Web Interface)
"""

import sys
from pathlib import Path
import streamlit as st
import torch

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import get_config, PathConfig, DeviceConfig
from utils.checkpoint import get_latest_checkpoint, load_dbn_checkpoint

# Streamlit Page Setup
st.set_page_config(
    page_title="DBN Digit Generation | Deep Learning Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", caption="MNIST Handwritten Digits", use_container_width=True)
    st.title("🧠 DBN Digit Generator")
    st.caption("College Mini Project | Deep Learning")
    st.markdown("---")
    
    st.subheader("🖥️ Hardware & Environment")
    st.info(f"**Execution Device:**\n{device_cfg.device_name}")
    st.markdown(f"**PyTorch Version:** `{torch.__version__}`")
    
    # Checkpoint status
    latest_ckpt = get_latest_checkpoint(paths.CHECKPOINT_DIR)
    if latest_ckpt is not None:
        st.success(f"**Active Checkpoint:**\n`{latest_ckpt.name}`")
    else:
        st.warning("No trained checkpoint found.\nGo to **Model Training** page to train.")

    st.markdown("---")
    st.markdown("### 📚 Navigation")
    st.markdown("""
    Use the sidebar pages to explore:
    - **1_Dataset_Explorer**: Inspect MNIST samples & statistics
    - **2_Model_Training**: Pretrain RBMs with live diagnostics
    - **3_Generate_Digits**: Probabilistically generate digits
    - **4_Real_vs_Generated**: Side-by-side visual comparison
    - **5_Evaluation**: Recognizability & pixel metrics
    - **6_About_and_Viva**: Project report, viva Q&A & script
    """)

# ----------------- MAIN CONTENT -----------------
st.title("🧠 Deep Belief Network for Synthetic Handwritten Digit Generation")
st.markdown("### *Stacked Restricted Boltzmann Machines with Contrastive Divergence & Directed Top-Down Sampling*")
st.markdown("---")

# Quick Metrics Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Dataset", value="MNIST (60k)", delta="28x28 Grayscale")
with col2:
    st.metric(label="Architecture", value="Stacked RBMs", delta="784 -> 500 -> 250 -> 100")
with col3:
    st.metric(label="Training Strategy", value="Greedy CD-k", delta="Layer-wise Pretraining")
with col4:
    st.metric(label="Generation Mechanism", value="MCMC + Ancestral", delta="Top-RBM Gibbs Sampling")

st.markdown("## 📖 1. Project Overview & Research Objective")
st.write("""
This project implements a complete, working deep generative model for handwritten digits from the ground up using **PyTorch**.
Rather than treating deep learning as an uninterpretable black box or relying on modern black-box GANs/VAEs, this project demonstrates **Geoffrey Hinton's foundational Deep Belief Network (DBN)** architecture.
The model learns hierarchical representations layer by layer through **Restricted Boltzmann Machines (RBMs)** and synthesizes novel handwritten digits via **alternating Gibbs sampling** in its top associative memory combined with **directed top-down ancestral generation**.
""")

st.markdown("## 🔬 2. Mathematical Framework")

tab1, tab2, tab3 = st.tabs(["1. Restricted Boltzmann Machine (RBM)", "2. Contrastive Divergence (CD-k)", "3. Probabilistic Generation in DBN"])

with tab1:
    st.markdown("""
    An RBM is an energy-based, undirected bipartite graphical model with visible units $\\mathbf{v} \\in \\{0, 1\\}^D$ and hidden units $\\mathbf{h} \\in \\{0, 1\\}^K$.
    
    **Energy Function:**
    $$E(\\mathbf{v}, \\mathbf{h}) = -\\mathbf{v}^T \\mathbf{W} \\mathbf{h} - \\mathbf{a}^T \\mathbf{v} - \\mathbf{b}^T \\mathbf{h} = -\\sum_{i} a_i v_i - \\sum_{j} b_j h_j - \\sum_{i,j} v_i W_{ij} h_j$$
    
    Because there are no intra-layer connections (visible-to-visible or hidden-to-hidden), the conditional distributions factorize completely:
    $$P(h_j = 1 \\mid \\mathbf{v}) = \\sigma\\left(b_j + \\sum_{i} v_i W_{ij}\\right)$$
    $$P(v_i = 1 \\mid \\mathbf{h}) = \\sigma\\left(a_i + \\sum_{j} W_{ij} h_j\\right)$$
    where $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ is the logistic sigmoid function.
    """)

with tab2:
    st.markdown("""
    The maximum likelihood gradient with respect to weight $W_{ij}$ is given by:
    $$\\frac{\\partial \\log P(\\mathbf{v})}{\\partial W_{ij}} = \\langle v_i h_j \\rangle_{\\text{data}} - \\langle v_i h_j \\rangle_{\\text{model}}$$
    
    Because calculating $\\langle \\cdot \\rangle_{\\text{model}}$ requires summing over $2^D$ states, **Hinton's Contrastive Divergence (CD-k)** approximates this by running $k$ alternating Gibbs sampling steps initialized at the data vector $\\mathbf{v}^{(0)}$:
    
    1. Positive phase: Sample $\\mathbf{h}^{(0)} \\sim P(\\mathbf{h} \\mid \\mathbf{v}^{(0)})$.
    2. Negative phase: Alternate $k$ steps: $\\mathbf{v}^{(t)} \\sim P(\\mathbf{v} \\mid \\mathbf{h}^{(t-1)})$, $\\mathbf{h}^{(t)} \\sim P(\\mathbf{h} \\mid \\mathbf{v}^{(t)})$.
    3. Update: $\\Delta W_{ij} = \\eta \\left( v_i^{(0)} P(h_j^{(0)}=1 \\mid \\mathbf{v}^{(0)}) - v_i^{(k)} P(h_j^{(k)}=1 \\mid \\mathbf{v}^{(k)}) \\right)$.
    """)

with tab3:
    st.markdown("""
    A stack of RBMs forms a **Deep Belief Network (DBN)**:
    - **Top layer (RBM 3)**: Operates as an *undirected associative memory*.
    - **Lower layers (RBM 2 and RBM 1)**: Act as a *directed generative belief network* with downward synthesis weights $\\mathbf{W}^T$.

    **Mathematically Justified Generative Sampling:**
    1. Initialize the visible state of the top RBM with random Bernoulli bits.
    2. Run alternating Gibbs sampling in RBM 3 for $M$ steps ($M \\ge 100$) until the Markov chain mixes to equilibrium on the joint prior $P(\\mathbf{h}_2, \\mathbf{h}_3)$.
    3. Perform **downward ancestral propagation**:
       - Pass $\\mathbf{h}_2$ downwards through RBM 2: $P(\\mathbf{h}_1 = 1 \\mid \\mathbf{h}_2) = \\sigma(\\mathbf{h}_2 \\mathbf{W}_2^T + \\mathbf{a}_2)$.
       - Pass $\\mathbf{h}_1$ downwards through RBM 1: $P(\\mathbf{v} = 1 \\mid \\mathbf{h}_1) = \\sigma(\\mathbf{h}_1 \\mathbf{W}_1^T + \\mathbf{a}_1)$.
    4. The output $\\mathbf{v}$ yields the synthetic $28 \\times 28$ grayscale digit image!
    """)

st.markdown("## 🔄 3. System Architecture & Workflow")
st.markdown("""
```
 +-------------------------------------------------------------------------+
 |                          MNIST Handwritten Dataset                      |
 |                     60,000 Images (28 x 28 Grayscale)                   |
 +-----------------------------------+-------------------------------------+
                                     | Normalization [0, 1] & Flatten (784)
                                     v
                       +---------------------------+
                       |    RBM 1: 784 -> 500      | (Trained via CD-1)
                       +-------------+-------------+
                                     | Hidden Probabilities P(h1|v)
                                     v
                       +---------------------------+
                       |    RBM 2: 500 -> 250      | (Trained via CD-1)
                       +-------------+-------------+
                                     | Hidden Probabilities P(h2|h1)
                                     v
                       +---------------------------+
                       |    RBM 3: 250 -> 100      | (Top Associative Memory)
                       +-------------+-------------+
                                     |
                       =============================
                       SYNTHETIC DIGIT GENERATION:
                       1. Alternating Gibbs Sampling in RBM 3 (M steps)
                       2. Top-down Directed Propagation:
                          RBM 3 -> RBM 2 -> RBM 1 -> Visible (784)
                       3. Reshape to (28, 28) Synthetic Grayscale Image
                       =============================
                                     |
                                     v
       +-----------------------------------------------------------+
       |   Evaluation: Visual Grid, Reconstruction MSE, Histogram  |
       |     & Independent MNIST CNN Classifier Recognizability    |
       +-----------------------------------------------------------+
```
""")

st.markdown("## 🚦 4. Live System Status")
stat_c1, stat_c2, stat_c3 = st.columns(3)

with stat_c1:
    mnist_path = paths.DATA_DIR / "MNIST"
    if mnist_path.exists():
        st.success("✅ **MNIST Dataset:** Downloaded & Available")
    else:
        st.info("ℹ️ **MNIST Dataset:** Will auto-download upon first training run.")

with stat_c2:
    if latest_ckpt is not None:
        st.success(f"✅ **Trained Checkpoint:** `{latest_ckpt.name}` ready")
    else:
        st.warning("⚠️ **Trained Checkpoint:** Not found. Train model on Page 2.")

with stat_c3:
    gen_files = list(paths.GENERATED_DIR.glob("*.png"))
    if gen_files:
        st.success(f"✅ **Generated Images:** {len(gen_files)} files ready")
    else:
        st.info("ℹ️ **Generated Images:** None generated yet. Visit Page 3.")

st.markdown("---")
st.markdown("👉 **Get Started:** Click **`2_Model_Training`** in the left sidebar to train the DBN, or **`3_Generate_Digits`** if a checkpoint is already present!")
