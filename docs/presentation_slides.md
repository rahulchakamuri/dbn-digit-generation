# College Project Presentation Slide Deck Outline

**Project Title:** Deep Belief Network for Synthetic Handwritten Digit Generation Using Stacked Restricted Boltzmann Machines  
**Degree:** B.Tech Artificial Intelligence and Machine Learning  
**Target Duration:** 10 to 15 Minutes (16 Slides)

---

## Slide 1: Title & Team Information
- **Title:** Deep Belief Network for Synthetic Handwritten Digit Generation
- **Subtitle:** Unsupervised Representation Learning and Probabilistic Sampling via Stacked Restricted Boltzmann Machines
- **Domain:** Deep Learning / Generative Modeling / Energy-Based Models
- **Dataset:** MNIST (60,000 $28 \times 28$ Grayscale Handwritten Digits)
- **Frameworks:** Python, PyTorch, Streamlit, Matplotlib

*Speaker Notes:* Introduce the project, team members, faculty guide, and acknowledge the importance of unsupervised learning in deep representations.

---

## Slide 2: Problem Statement & Motivation
- **The Challenge:** How can a neural network learn the complex, multi-modal probability distribution of handwritten numerals without relying on modern black-box GANs or continuous VAEs?
- **Why DBNs?**
  - Historical milestone: Launched the modern deep learning revolution in 2006 (Geoffrey Hinton).
  - Energy-based foundation: Models probabilities explicitly through physical Boltzmann distributions.
  - Transparent & interpretable: Layer-wise feature abstraction and mathematically grounded MCMC sampling.
- **Hardware Barrier:** Most modern generative models require high-end GPUs; our project is optimized to run smoothly on everyday laptop CPUs.

---

## Slide 3: Project Objectives
1. Implement pure PyTorch Bernoulli Restricted Boltzmann Machines from foundational physics/energy equations.
2. Develop vectorized Contrastive Divergence ($CD_k$) training.
3. Stack three RBMs into a Deep Belief Network and execute greedy layer-wise pretraining.
4. Formulate an authentic generative procedure: top-RBM Gibbs sampling + downward directed ancestral synthesis.
5. Benchmark real vs. generated images through statistical distributions and external CNN recognizability.
6. Build a modern Streamlit web dashboard for interactive demonstrations.

---

## Slide 4: Literature Survey
- **Hinton, Osindero, & Teh (2006) - *A Fast Learning Algorithm for Deep Belief Nets*:**
  - Proved that adding RBM layers monotonically increases a variational lower bound on data log-likelihood.
  - Solved the vanishing gradient problem that previously plagued deep networks.
- **Hinton (2002) - *Training Products of Experts by Minimizing Contrastive Divergence*:**
  - Replaced intractable infinite Markov chains with $k$-step Gibbs updates initialized at data vectors.
- **LeCun et al. (1998) - *Gradient-Based Learning Applied to Document Recognition*:**
  - Standardized the MNIST benchmark used across academic deep learning research.

---

## Slide 5: System Architecture

```
                    +-----------------------------+
                    |    Hidden Units (h3)        |  (100 units / Demo: 64)
                    |     Layer 3 Representation   |
                    +--------------+--------------+
                                   ^
                 Top Undirected    |  Symmetric Weights W3, Biases a3, b3
                 Associative RBM   v  (Alternating Gibbs Sampling MCMC)
                    +--------------+--------------+
                    |    Hidden Units (h2)        |  (250 units / Demo: 128)
                    |     Layer 2 Representation   |
                    +--------------+--------------+
                                   |
                  Directed         |  Downward Generative Weights W2^T, a2
                  Belief Net       v
                    +--------------+--------------+
                    |    Hidden Units (h1)        |  (500 units / Demo: 256)
                    |     Layer 1 Representation   |
                    +--------------+--------------+
                                   |
                  Directed         |  Downward Generative Weights W1^T, a1
                  Belief Net       v
                    +--------------+--------------+
                    |    Visible Units (v0)       |  (784 units = 28x28)
                    |     Synthesized Pixels      |
                    +-----------------------------+
```

---

## Slide 6: RBM Mathematical Foundations
- **Energy Function:**
  $$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h}$$
- **Joint Boltzmann Distribution:**
  $$P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{Z}, \quad Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$
- **Intractability of Partition Function $Z$:**
  - Requires summing over $2^{784 + 500} \approx 2^{1284}$ states (NP-hard).
- **Factorized Conditional Independence:**
  $$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_i v_i W_{ij}\right)$$
  $$P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_j W_{ij} h_j\right)$$

---

## Slide 7: Contrastive Divergence ($CD_k$)
- **Log-Likelihood Gradient:**
  $$\frac{\partial \ln P(\mathbf{v})}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$
- **Hinton's CD Approximation:**
  - Positive phase: driven by real training vector $\mathbf{v}^{(0)}$.
  - Negative phase: $k$ steps of alternating Gibbs updates: $\mathbf{v}^{(0)} \to \mathbf{h}^{(0)} \to \mathbf{v}^{(1)} \to \mathbf{h}^{(1)} \dots \to \mathbf{v}^{(k)} \to \mathbf{h}^{(k)}$.
- **PyTorch Surrogate Free Energy Formulation:**
  $$F(\mathbf{v}) = -\mathbf{a}^T \mathbf{v} - \sum_j \text{softplus}\left(\mathbf{v} \mathbf{W}_{*j} + b_j\right)$$
  $$\mathcal{L} = \text{mean}\left(F(\mathbf{v}^{(0)}) - F(\mathbf{v}^{(k)}_{\text{detached}})\right)$$

---

## Slide 8: Greedy Layer-Wise Pretraining
- **Step 1:** Train RBM 1 directly on raw normalized 784-dimensional pixel vectors.
- **Step 2:** Freeze RBM 1; propagate inputs to compute continuous activations: $\mathbf{h}_1 = P(\mathbf{h}_1 = 1 \mid \mathbf{v}_0)$.
- **Step 3:** Train RBM 2 using $\mathbf{h}_1$ as visible input vectors.
- **Step 4:** Freeze RBM 2; propagate to compute: $\mathbf{h}_2 = P(\mathbf{h}_2 = 1 \mid \mathbf{h}_1)$.
- **Step 5:** Train RBM 3 using $\mathbf{h}_2$ as visible inputs.
- **Advantage:** Each layer is optimized in its own convex-like sub-problem, avoiding vanishing gradients.

---

## Slide 9: Probabilistic Digit Generation
- **Crucial Distinction:** DBN generation does NOT pass arbitrary noise through the bottom layer!
- **Phase 1: Top-Level MCMC Mixing**
  - Initialize top associative RBM with random Bernoulli bits: $\mathbf{h}_2^{(0)} \sim \text{Bernoulli}(0.5)$.
  - Run alternating Gibbs sampling for $M = 100$ steps:
    $$\mathbf{h}_3^{(t)} \sim P(\mathbf{h}_3 \mid \mathbf{h}_2^{(t-1)}), \quad \mathbf{h}_2^{(t)} \sim P(\mathbf{h}_2 \mid \mathbf{h}_3^{(t)})$$
  - Chain settles into equilibrium on the joint prior $P(\mathbf{h}_2, \mathbf{h}_3)$.
- **Phase 2: Directed Ancestral Downward Propagation**
  - Project downward: $\mathbf{h}_1 = \sigma(\mathbf{h}_2^{(M)} \mathbf{W}_2^T + \mathbf{a}_2)$.
  - Project to visible: $\mathbf{v}_0 = \sigma(\mathbf{h}_1 \mathbf{W}_1^T + \mathbf{a}_1)$.
  - Reshape continuous visible probabilities into $(28, 28)$ synthetic image.

---

## Slide 10: Experimental Configuration
- **Dataset:** MNIST (6,000 active training samples in Demo Mode, 60,000 available).
- **Optimizer:** SGD with Momentum ($\gamma = 0.5$) and Weight Decay ($\lambda = 0.0001$).
- **Learning Rate:** $\eta = 0.05$.
- **Batch Size:** 64.
- **Hardware Environment:** Intel CPU (No GPU required).
- **Execution Speed:** Complete 3-RBM pretraining in **10.48 seconds**.

---

## Slide 11: Experimental Results & Convergence

| Layer | Architecture | Epochs | Initial Recon MSE | Final Recon MSE | % Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RBM 1** | $784 \to 256$ | 3 | 0.0574 | **0.0298** | **48.1%** |
| **RBM 2** | $256 \to 128$ | 3 | 0.0849 | **0.0250** | **70.6%** |
| **RBM 3** | $128 \to 64$ | 3 | 0.1064 | **0.0404** | **62.0%** |

- **Reconstruction MSE:** Steady reduction across all layers confirms representation learning.
- **Pixel Sparsity:** Real MNIST background is 81.7% zeros; generated digits preserve 74.4% background sparsity.

---

## Slide 12: Independent CNN Recognizability
- **External Evaluation:** Independent 2-layer CNN trained on MNIST (never used during DBN pretraining).
- **Recognizability Rate:** **44.0%** of synthetic samples classified with $\ge 70\%$ confidence.
- **Average Confidence:** **65.3%** across generated digits.
- **Digit Distribution:** Successfully generated diverse digits across multiple numeral classes (0, 1, 2, 3, 4, 5, 7, 8).
- **Academic Disclaimer:** Classifier confidence confirms feature alignment with MNIST stroke patterns; it is not a direct substitute for human aesthetic realism.

---

## Slide 13: Architectural Comparison: DBN vs GAN vs VAE
- **DBN:** Energy-based graphical model; layer-wise pretraining; MCMC Gibbs sampling; highly stable; runs on CPU.
- **GAN:** Minimax adversarial game; single-pass generation; prone to mode collapse; requires GPU.
- **VAE:** Variational inference; Gaussian latent prior; single-pass decoder; produces blurrier images.

---

## Slide 14: Limitations & Observed Characteristics
1. **Soft Margins & Diffuse Halos:** Visible output is the sigmoid expectation $P(v_i = 1 \mid \mathbf{h})$, creating slight grayscale softness near stroke borders.
2. **Mode Mixing / Chimeric Digits:** During MCMC sampling, transitions between energy basins can produce hybrid numerals (e.g. blend of a 3 and 8).
3. **MCMC Mixing Time:** Exploring all 10 modes requires adequate Gibbs burn-in iterations.

---

## Slide 15: Conclusion & Future Scope
- **Conclusion:**
  - Successfully designed, implemented, and verified a 3-layer Deep Belief Network for handwritten digit generation.
  - Verified 100% test pass rate on automated PyTest suite (13/13 tests).
  - Delivered an interactive Streamlit GUI with one-click training and PNG/ZIP downloads.
- **Future Work:**
  - Implement Persistent Contrastive Divergence (PCD) for improved mixing.
  - Add Gaussian-Bernoulli units for continuous color images (CIFAR-10).
  - Clamp one-hot label vectors to the top layer for class-steered digit generation.

---

## Slide 16: Viva Voce & Demonstration
- **Live Demo:** `streamlit run app.py` (or double-click `run_app.bat`).
- **Thank you! Questions and comments are welcomed.**
