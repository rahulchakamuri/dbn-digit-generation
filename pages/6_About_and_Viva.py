"""
Streamlit Page 6: Project Documentation, Viva Preparation, and Technical Review
Complete academic reference for college viva, presentation script, and 25 viva Q&As.
"""

import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="About & Viva Prep | DBN", page_icon="🎓", layout="wide")

st.title("🎓 Project Academic Reference & Viva Preparation Guide")
st.markdown("### *Deep Belief Network for Synthetic Handwritten Digit Generation*")
st.markdown("---")

tab_about, tab_pitch, tab_script, tab_comp, tab_viva = st.tabs([
    "📑 Project Overview & Report",
    "⏱️ 2-Minute Elevator Pitch",
    "🎤 5-Minute Demonstration Script",
    "⚖️ DBN vs GAN vs VAE vs Diffusion",
    "❓ 25 Viva Questions & Answers"
])

# ----------------- TAB 1: ABOUT & REPORT -----------------
with tab_about:
    st.header("1. Academic Abstract")
    st.markdown("""
    This project presents the design, mathematical formulation, implementation, and empirical evaluation of a **Deep Belief Network (DBN)** for synthetic handwritten digit generation using the MNIST dataset.
    A DBN is constructed by stacking three **Restricted Boltzmann Machines (RBMs)** ($784 \\to 500 \\to 250 \\to 100$) trained greedily layer-by-layer via **Contrastive Divergence ($CD_1$)**.
    Once pretrained, generative sampling is executed by running alternating **Gibbs sampling** in the top undirected associative memory layer to reach equilibrium on the learned joint prior, followed by an **ancestral downward pass** through the lower directed generative belief network to synthesize $28 \\times 28$ grayscale images.
    """)

    st.header("2. Problem Statement & Motivation")
    st.markdown("""
    While modern deep generative modeling is dominated by Generative Adversarial Networks (GANs) and Denoising Diffusion Models, these models function primarily as continuous gradient-based mappings that often suffer from mode collapse, unstable min-max dynamics, and high computational barriers requiring dedicated GPUs.
    In contrast, **Deep Belief Networks (Hinton et al., 2006)** ignited the modern deep learning revolution by introducing **probabilistic layer-wise pretraining**. Understanding and implementing energy-based models, Markov Chain Monte Carlo (MCMC), and directed downward synthesis provides deep, rigorous insights into probabilistic machine learning.
    """)

    st.header("3. Key Objectives")
    st.markdown("""
    1. Implement a modular, pure **PyTorch** Restricted Boltzmann Machine with Bernoulli units.
    2. Implement **Contrastive Divergence ($CD_k$)** with surrogate free energy formulation.
    3. Construct a stacked **Deep Belief Network** with greedy layer-wise feature propagation.
    4. Formulate a **mathematically valid generation procedure** (top-level MCMC + downward directed synthesis).
    5. Evaluate synthetic outputs against real digits using statistical pixel distributions, reconstruction MSE, and an independent CNN classifier.
    6. Deliver an interactive **Streamlit** demonstration interface with zero-dependency laptop CPU support.
    """)

# ----------------- TAB 2: ELEVATOR PITCH -----------------
with tab_pitch:
    st.header("⏱️ 2-Minute Project Elevator Pitch")
    st.info("""
    **Speak this with confidence during the initial 2-minute project introduction:**
    
    \"Good morning, respected professors and panel members.
    
    Our project is titled **'Deep Belief Network for Synthetic Handwritten Digit Generation Using Stacked Restricted Boltzmann Machines'**.
    
    The core problem we address is: **How can a probabilistic neural network learn the complex multi-modal distribution of handwritten digits and generate new, unseen digits from scratch without relying on modern black-box GANs?**
    
    To solve this, we implemented **Geoffrey Hinton's foundational Deep Belief Network** architecture using PyTorch. Our architecture stacks three Restricted Boltzmann Machines. The training proceeds through **greedy layer-wise pretraining**:
    - The first RBM learns primitive stroke features directly from the 784 flattened MNIST pixels.
    - The second and third RBMs learn higher-order stroke conjunctions and digit topologies from the hidden representations of the previous layers.
    
    For synthetic digit generation, we do not simply inject arbitrary noise. Instead, we perform **alternating Gibbs sampling** in the top associative memory layer for 100 steps until the Markov chain reaches stationary equilibrium. Then, we perform an **ancestral downward pass** through the lower directed sigmoid layers to project the latent state back into visible pixel space.
    
    We evaluated the system through three complementary methods:
    1. **End-to-end Reconstruction MSE**, which dropped steadily during training.
    2. **Pixel Intensity Histograms**, confirming the model preserved the ~80% background sparsity of MNIST.
    3. An **Independent CNN Classifier**, which serves as an objective external proxy to quantify digit recognizability.
    
    The entire application is deployed with an interactive **Streamlit GUI**, supporting both a fast Laptop Demo Mode and a Full Academic Training Mode. Thank you, and we welcome your questions.\"
    """)

# ----------------- TAB 3: DEMO SCRIPT -----------------
with tab_script:
    st.header("🎤 5-Minute Viva Demonstration Script")
    st.markdown("""
    Follow these exact steps during your practical laboratory review:
    
    1. **Minute 0:00 - 1:00 | Launch & System Architecture:**
       - Open your terminal and run: `streamlit run app.py`.
       - Show the **Home Dashboard**. Point out the active execution device (CPU/GPU) and the architecture block diagram showing the 3 stacked RBMs.
    
    2. **Minute 1:00 - 2:00 | Dataset Explorer (Page 1):**
       - Navigate to **`1_Dataset_Explorer`**.
       - Explain the $28 \\times 28 = 784$ pixel flattening and the continuous $[0, 1]$ normalization.
       - Show the pixel histogram: explain that ~80% of pixels are zeros (background) and explain why Bernoulli RBMs are well-suited for this distribution.
    
    3. **Minute 2:00 - 3:00 | Layer-Wise Training Diagnostics (Page 2):**
       - Navigate to **`2_Model_Training`**.
       - Point to the training curves. Explain that Contrastive Divergence ($CD_1$) reduced the reconstruction error epoch-by-epoch across all 3 RBM layers.
       - Highlight the checkpointing system: pre-trained weights are persisted so retraining is not mandatory upon every execution.
    
    4. **Minute 3:00 - 4:00 | Live Digit Generation (Page 3):**
       - Navigate to **`3_Generate_Digits`**.
       - Choose 25 samples, 100 Gibbs steps, and click **Generate Synthetic Digits**.
       - Explain the two-phase sampling: MCMC mixing in the top RBM, followed by downward directed ancestral propagation.
       - Demonstrate the one-click **PNG Grid** and **ZIP** download features.
    
    5. **Minute 4:00 - 5:00 | Evaluation & Viva Defense (Pages 4 & 5):**
       - Navigate to **`4_Real_vs_Generated`** and **`5_Evaluation`**.
       - Show the side-by-side comparison. Point out genuine structural similarities and explain the characteristic soft/diffuse edges of RBM Bernoulli expectations.
       - Show the external CNN recognizability score and conclude with the architectural differences between DBNs and GANs.
    """)

# ----------------- TAB 4: ARCHITECTURAL COMPARISON -----------------
with tab_comp:
    st.header("⚖️ Comparative Analysis: DBN vs. GAN vs. VAE vs. Diffusion")
    st.markdown("""
    | Architectural Feature | Deep Belief Network (DBN) | Generative Adversarial Network (GAN) | Variational Autoencoder (VAE) | Denoising Diffusion (DDPM) |
    | :--- | :--- | :--- | :--- | :--- |
    | **Foundational Paradigm** | Energy-Based Graphical Model | Game-Theoretic Minimax Optimization | Variational Bayesian Inference | Stochastic Reverse Differential Equations |
    | **Layer Type** | Undirected Top RBM + Directed Lower Layers | Feedforward/Convolutional (Generator & Discriminator) | Encoder-Decoder Networks | Multi-step U-Net Denoisers |
    | **Training Strategy** | Greedy Layer-Wise Pretraining ($CD_k$) | Joint Adversarial Gradient Descent | Evidence Lower Bound (ELBO) Optimization | Score Matching / MSE on Injected Gaussian Noise |
    | **Sampling Mechanism** | MCMC (Gibbs Sampling) in Top Layer + Downward Pass | Single-pass forward pass from Gaussian prior | Single-pass forward pass through decoder | Multi-step iterative Langevin denoising |
    | **Training Stability** | Highly stable (convex-like per layer) | Prone to mode collapse and oscillation | Stable (analytical gradient with reparameterization) | Extremely stable, high computational cost |
    | **Hardware Requirement** | Runs easily on standard laptop CPU | Requires dedicated GPU | Runs on CPU/GPU | Requires powerful GPU clusters |
    | **Sample Sharpness** | Moderately soft (Bernoulli expectation) | Extremely sharp | Often blurry (Gaussian likelihood penalty) | State-of-the-art photorealistic sharpness |
    """)

# ----------------- TAB 5: 25 VIVA Q&A -----------------
with tab_viva:
    st.header("❓ 25 Essential Viva Questions and Detailed Answers")

    viva_qa = [
        ("1. What is a Restricted Boltzmann Machine (RBM)?",
         "An RBM is an energy-based, undirected bipartite graphical model consisting of two layers: a visible layer representing input data and a hidden layer representing latent features. It is 'restricted' because there are strictly no connections between units in the same layer (no visible-to-visible or hidden-to-hidden connections)."),
        
        ("2. Why is the restriction in an RBM so important mathematically?",
         "The absence of intra-layer connections makes the hidden units conditionally independent given the visible units, and vice versa. This allows exact, parallel calculation of conditional probabilities: P(h|v) = prod P(h_j|v) and P(v|h) = prod P(v_i|h), making block Gibbs sampling computationally efficient."),
        
        ("3. What is the Energy Function of an RBM?",
         "For visible vector v and hidden vector h, the energy is defined as: E(v, h) = -v^T W h - a^T v - b^T h, where W is the weight matrix, a is the visible bias vector, and b is the hidden bias vector. Configurations with lower energy have higher probability under the Boltzmann distribution."),
        
        ("4. What is the joint probability distribution of an RBM?",
         "P(v, h) = exp(-E(v, h)) / Z, where Z is the partition function: Z = sum_{v, h} exp(-E(v, h)). Computing Z is intractable for large networks because it requires summing over 2^(visible_dim + hidden_dim) configurations."),
        
        ("5. What is Free Energy in an RBM?",
         "The free energy F(v) is the effective energy of a visible vector after summing over all hidden states: exp(-F(v)) = sum_h exp(-E(v, h)). Mathematically: F(v) = -a^T v - sum_j softplus(v W_{*,j} + b_j). It enables the analytical computation of log-likelihood gradients."),
        
        ("6. What is Contrastive Divergence (CD) and why is it used?",
         "Computing the exact gradient of the log-likelihood requires calculating expectations under the model distribution, which requires infinitely long MCMC runs. Hinton's Contrastive Divergence (CD-k) approximates this by running Gibbs sampling for only k steps (typically k=1) initialized at the empirical data vector v^(0)."),
        
        ("7. What is the difference between the positive phase and negative phase in CD?",
         "The positive phase computes <v_i h_j>_data using the real training sample v^(0) and its hidden activation probabilities. The negative phase computes <v_i h_j>_model using reconstructed visible and hidden states after k steps of Gibbs sampling. The parameter update shifts the energy down on real data and up on model-hallucinated reconstructions."),
        
        ("8. Why do we use probabilities instead of binary samples in the negative phase?",
         "Using continuous probabilities P(h_k = 1 | v_k) instead of stochastic binary samples for the hidden units in the negative phase significantly reduces sampling variance and accelerates training convergence, as recommended by Geoffrey Hinton."),
        
        ("9. What is a Deep Belief Network (DBN)?",
         "A Deep Belief Network is a multi-layer deep neural network formed by stacking multiple RBMs. The top two layers form an undirected associative memory (an RBM), while the lower layers form a directed acyclic generative belief network with top-down connections."),
        
        ("10. What is Greedy Layer-Wise Pretraining?",
         "It is an unsupervised training strategy introduced by Geoffrey Hinton in 2006. Each RBM layer is trained sequentially: RBM 1 trains directly on raw data; then its hidden activations serve as the 'data' to train RBM 2; then RBM 2's activations train RBM 3. Each layer guarantees an improvement in the variational lower bound of log-likelihood."),
        
        ("11. How does image generation work in a DBN?",
         "Generation does NOT pass random noise forward. Instead, we run alternating Gibbs sampling (MCMC) in the top-level RBM for M steps to sample an equilibrium feature state from the learned joint prior. That state is then passed downwards through the lower layers via directed conditional distributions P(v_{l-1} | h_l) = sigmoid(h_l W_l^T + a_l) to produce the visible 28x28 image."),
        
        ("12. What is Gibbs Sampling?",
         "Gibbs sampling is a Markov Chain Monte Carlo (MCMC) technique for generating observations from a multivariate probability distribution when direct sampling is difficult. It alternates between conditionally updating hidden units given visible units, and visible units given hidden units."),
        
        ("13. Why does the top layer of a DBN act as an associative memory?",
         "Because its connections are bidirectional and symmetric (undirected). It can complete partial patterns, settle into attractor states corresponding to learned digit classes, and generate coherent high-level feature combinations."),
        
        ("14. How are MNIST images preprocessed for Bernoulli RBMs?",
         "Each 28x28 grayscale image is scaled from [0, 255] to [0.0, 1.0] and flattened into a 784-dimensional vector. For Bernoulli RBMs, pixel values are treated as the probability of the visible unit being active (1), which aligns with the logistic sigmoid output."),
        
        ("15. Why do some generated digits appear blurry or have fuzzy borders?",
         "In a Bernoulli RBM/DBN, the visible output is the conditional expectation P(v_i = 1 | h) = sigmoid(z). Unlike GANs which force hard binary or sharp discriminator boundaries, the DBN reports activation probabilities. Pixels with stroke uncertainty near digit boundaries naturally receive intermediate gray values."),
        
        ("16. What is Mode Mixing in DBN generation?",
         "The Gibbs sampler navigates a multimodal energy landscape where each digit represents an energy valley. If the Markov chain transitions between two basins (e.g., between the basin for '3' and '8'), it can sample intermediate states, producing hybrid or ambiguous digits."),
        
        ("17. What is the role of the independent CNN classifier in this project?",
         "It acts as an objective, external evaluator to assess digit recognizability. It calculates whether the synthesized pixels trigger standard digit class features. Importantly, it was trained separately on MNIST and was never used during DBN pretraining."),
        
        ("18. Does high classifier confidence mean the generated image is visually perfect?",
         "No. A classifier looks for specific spatial correlations and stroke patterns. It can assign 95% confidence to a noisy or slightly deformed digit. Classifier confidence is an objective proxy metric for feature alignment, not human-perceived aesthetic realism."),
        
        ("19. Why did deep learning transition from DBNs to GANs and Diffusion models?",
         "DBNs rely on MCMC sampling, which can take time to mix in high dimensions, and scaling undirected bipartite models to high-resolution color images is difficult. GANs and Diffusion models use continuous convolutions and direct gradient backpropagation, scaling much better to megapixel resolutions."),
        
        ("20. What is the significance of DBNs in the history of Deep Learning?",
         "In 2006, Hinton, Osindero, and Teh published 'A Fast Learning Algorithm for Deep Belief Nets'. Before this paper, training deep architectures was considered impossible due to vanishing gradients. DBN layer-wise pretraining proved that deep models could be trained effectively, kicking off the modern deep learning revolution."),
        
        ("21. What happens if learning rate is set too high in RBM training?",
         "The weights can diverge rapidly, leading to saturated hidden units (sigmoid outputs pinned at 0 or 1), gradient explosion, and catastrophic failure to reconstruct input vectors."),
        
        ("22. What is weight decay and why is it used in RBMs?",
         "Weight decay applies an L2 penalty on the weights (lambda * W). It prevents weights from growing excessively large, which keeps the sigmoid activations in their sensitive, non-saturated linear region and regularizes the energy surface."),
        
        ("23. What is momentum in Contrastive Divergence?",
         "Momentum adds a fraction (e.g., 0.5 to 0.9) of the previous weight update vector to the current step. It dampens oscillations in high-curvature directions and accelerates learning across flat energy plateaus."),
        
        ("24. Can a DBN perform classification as well as generation?",
         "Yes. By adding a softmax label layer to the top associative RBM, a DBN can be fine-tuned with backpropagation for supervised classification. In our project, we focused on its generative capability."),
        
        ("25. How would you improve this project in future work?",
         "Future enhancements could include: (1) implementing persistent Contrastive Divergence (PCD) to improve Gibbs chain mixing; (2) adding Gaussian-Bernoulli visible units for continuous natural images; (3) fine-tuning the entire network with the wake-sleep algorithm; and (4) conditioning the top layer on one-hot class vectors for label-steered digit generation.")
    ]

    for q, a in viva_qa:
        with st.expander(f"📌 {q}", expanded=False):
            st.write(a)
