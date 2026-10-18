# College Mini-Project Report

## DEEP BELIEF NETWORK FOR SYNTHETIC HANDWRITTEN DIGIT GENERATION USING STACKED RESTRICTED BOLTZMANN MACHINES

**Course:** B.Tech Artificial Intelligence and Machine Learning  
**Subject:** Deep Learning Laboratory / Mini-Project  
**Academic Year:** 2026-2027  

---

## ABSTRACT

Synthetic handwritten digit generation represents a classical benchmark in unsupervised representation learning and deep generative modeling. While modern research has largely shifted toward continuous gradient-based Generative Adversarial Networks (GANs) and Denoising Diffusion Probabilistic Models (DDPMs), these approaches frequently obscure the foundational probabilistic principles of energy-based learning, latent equilibrium distributions, and Markov Chain Monte Carlo (MCMC) sampling.

This project implements a complete, mathematically grounded **Deep Belief Network (DBN)** composed of three stacked **Restricted Boltzmann Machines (RBMs)** ($784 \to 500 \to 250 \to 100$) trained on the MNIST handwritten digit benchmark using **PyTorch**. The network utilizes **greedy layer-wise pretraining** governed by **Contrastive Divergence ($CD_1$)** with an analytical surrogate free-energy formulation. For image synthesis, we implement a probabilistically justified generation strategy: alternating Gibbs sampling is executed in the top-level undirected associative memory layer to achieve stationary equilibrium on the latent digit prior, followed by an ancestral downward pass through the lower directed generative belief network to reconstruct continuous $28 \times 28$ grayscale pixel intensities.

Quantitative and qualitative evaluations demonstrate steady reduction in layer-wise reconstruction error, preservation of the characteristic ~80% MNIST background sparsity, and substantial digit recognizability verified via an independent CNN classifier. An interactive **Streamlit** dashboard provides a responsive web application supporting real-time pretraining, diagnostic visualization, interactive digit generation, and side-by-side comparative analysis on standard laptop CPUs.

---

## 1. INTRODUCTION

Deep generative models aim to approximate an unknown underlying probability distribution $P_{\text{data}}(\mathbf{x})$ given a finite set of empirical training observations. In 2006, Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh published the seminal paper *"A Fast Learning Algorithm for Deep Belief Nets"*, demonstrating that deep multi-layer neural architectures could be successfully trained in an unsupervised manner using greedy layer-by-layer pretraining. This breakthrough revived deep neural network research and laid the groundwork for modern artificial intelligence.

Understanding the mechanics of Restricted Boltzmann Machines (RBMs) and Deep Belief Networks (DBNs) provides undergraduate engineering students with essential theoretical grounding in energy-based models, partition function intractability, contrastive divergence approximations, and stochastic MCMC mixing.

---

## 2. PROBLEM STATEMENT & OBJECTIVES

### 2.1 Problem Statement
Develop an end-to-end, reproducible deep learning system capable of learning the high-dimensional probability distribution of handwritten digits from the MNIST dataset and generating novel, synthetic digit images using a Deep Belief Network, while operating efficiently on standard consumer laptop hardware without requiring dedicated GPU acceleration.

### 2.2 Project Objectives
1. **Mathematical Implementation:** Develop a pure PyTorch implementation of the Bernoulli Restricted Boltzmann Machine and Deep Belief Network from foundational energy equations.
2. **Contrastive Divergence Pipeline:** Implement the Contrastive Divergence ($CD_k$) training algorithm with SGD momentum and $L_2$ weight decay.
3. **Layer-Wise Pretraining:** Sequentially pretrain three stacked RBMs ($784 \to 500 \to 250 \to 100$ or demo scale $784 \to 256 \to 128 \to 64$), transforming empirical pixel observations into hierarchical latent representations.
4. **Probabilistic Image Generation:** Formulate and verify an authentic generative procedure: top-RBM Gibbs sampling to equilibrium followed by directed top-down ancestral synthesis.
5. **Rigorous Evaluation:** Quantitatively evaluate the model using end-to-end reconstruction mean squared error (MSE), pixel distribution histograms, and an independent evaluation classifier.
6. **Student-Friendly Web Interface:** Deploy an interactive Streamlit graphical interface enabling one-click training, interactive sample generation, parameter customization, and PNG/ZIP exporting.

---

## 3. LITERATURE SURVEY

| Author & Year | Title / Contribution | Relevance to Project | Key Limitation Addressed |
| :--- | :--- | :--- | :--- |
| **Hinton, Osindero, & Teh (2006)** | *"A Fast Learning Algorithm for Deep Belief Nets"* | Introduced layer-wise greedy pretraining and the top-RBM associative memory formulation. | Solved vanishing gradient problem in deep networks. |
| **Hinton (2002)** | *"Training Products of Experts by Minimizing Contrastive Divergence"* | Formulated the $CD_k$ approximation to avoid infinite Markov chain mixing during training. | Replaced intractable partition function calculation with $k$-step Gibbs updates. |
| **Tieleman (2008)** | *"Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient"* | Proposed Persistent Contrastive Divergence (PCD) maintaining continuous chains across mini-batches. | Addressed potential bias in standard $CD_1$ when restarting chains from data. |
| **Goodfellow et al. (2014)** | *"Generative Adversarial Networks"* | Proposed adversarial game-theoretic continuous image generation. | Contrast to DBN: GANs lack explicit probabilistic energy formulations and suffer from mode collapse. |

---

## 4. SYSTEM REQUIREMENTS SPECIFICATION (SRS)

### 4.1 Functional Requirements
- **FR1 (Dataset Management):** Automated download, verification, normalization ($[0.0, 1.0]$), and flattening ($784$-dim) of the MNIST dataset.
- **FR2 (Layer-Wise Training):** Configurable training of RBM 1, RBM 2, and RBM 3 with real-time loss tracking.
- **FR3 (Model Checkpointing):** Persistence of model state dictionaries, architectural configurations, and loss histories.
- **FR4 (Probabilistic Generation):** Configurable Gibbs sampling steps (20 to 500) and sample counts (10, 25, 50, 100) with reproducible seed control.
- **FR5 (Evaluation & Comparison):** Calculation of reconstruction MSE, background sparsity ratios, and external classifier recognizability scores.
- **FR6 (Data Export):** Exporting generated synthetic digits as PNG image files and consolidated ZIP packages.

### 4.2 Non-Functional Requirements
- **Portability:** Operates transparently on Windows PowerShell and POSIX shells.
- **Hardware Independence:** Automatically defaults to CPU execution without requiring NVIDIA CUDA drivers.
- **Modularity:** Strict separation of configuration, data loading, core models, visualization, tests, and web UI.
- **Performance:** Fast Demo Mode executes complete 3-layer pretraining in under 15 seconds on a standard dual-core/quad-core laptop CPU.

---

## 5. SYSTEM ARCHITECTURE & METHODOLOGY

### 5.1 Model Architecture
The Deep Belief Network consists of three stacked Bernoulli RBMs:
- **Input Dimension:** 784 visible units ($28 \times 28$ grayscale pixels).
- **RBM 1:** 784 visible units $\to$ 500 hidden units (or 256 for Demo Mode).
- **RBM 2:** 500 visible units $\to$ 250 hidden units (or 128 for Demo Mode).
- **RBM 3 (Top Layer):** 250 visible units $\to$ 100 hidden units (or 64 for Demo Mode).

### 5.2 Mathematical Formulation
1. **Energy Function:**
   $$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h}$$
2. **Conditional Distributions:**
   $$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_i v_i W_{ij}\right), \quad P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_j W_{ij} h_j\right)$$
3. **Surrogate Free Energy Loss:**
   $$F(\mathbf{v}) = -\mathbf{a}^T \mathbf{v} - \sum_j \text{softplus}(\mathbf{v} \mathbf{W}_{*j} + b_j)$$
   $$\mathcal{L}_{\text{CD}} = \frac{1}{B} \sum_{b=1}^B \left[ F(\mathbf{v}^{(0)}) - F(\mathbf{v}^{(k)}_{\text{detached}}) \right]$$
4. **Ancestral Downward Synthesis:**
   Following top-RBM Gibbs sampling to equilibrium $\mathbf{h}_2 \sim P(\mathbf{h}_2, \mathbf{h}_3)$, activations propagate downwards:
   $$\mathbf{h}_1 = \sigma(\mathbf{h}_2 \mathbf{W}_2^T + \mathbf{a}_2), \quad \mathbf{v}_0 = \sigma(\mathbf{h}_1 \mathbf{W}_1^T + \mathbf{a}_1)$$

---

## 6. IMPLEMENTATION DETAILS

The implementation is structured into clean, modular Python packages:
- `config/config.py`: Centralized configuration management with hardware detection and mode presets.
- `models/rbm.py`: PyTorch `nn.Module` for the Restricted Boltzmann Machine with vectorized CD updates.
- `models/dbn.py`: Stacked Deep Belief Network with upward feature extraction, downward directed synthesis, and Gibbs sampling.
- `models/train.py`: Pretraining orchestrator supporting live callbacks and checkpoint persistence.
- `models/generate.py`: Standalone CLI generation script producing individual PNGs and grid collages.
- `models/evaluate.py`: Diagnostic suite computing reconstruction error, pixel statistics, and recognizability.
- `models/classifier.py`: Independent 2-stage CNN classifier for objective recognizability scoring.
- `utils/`: Data loading pipelines, visualization tools, and checkpoint loaders.
- `pages/`: Streamlit multi-page interactive web application.

---

## 7. EXPERIMENTAL RESULTS & DISCUSSION

### 7.1 Pretraining Convergence
Training was executed on an Intel CPU with 6,000 active training samples across 3 epochs per layer:

| Layer | Architecture | Initial Loss | Final Loss | Initial Recon MSE | Final Recon MSE | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RBM 1** | $784 \to 256$ | -23.7664 | -17.5083 | 0.0574 | **0.0298** | Converged |
| **RBM 2** | $256 \to 128$ | -4.9481 | -8.4019 | 0.0849 | **0.0250** | Converged |
| **RBM 3** | $128 \to 64$ | -1.9817 | -6.9153 | 0.1064 | **0.0404** | Converged |

**Observation:** Across all three stacked RBMs, reconstruction MSE decreased consistently, demonstrating that each layer successfully captured and reconstructed the feature distribution of the previous layer. Total pretraining elapsed time was **10.48 seconds**.

### 7.2 Statistical Pixel Distribution
Evaluating 25 generated samples against 25 real MNIST images:
- **Real Mean Pixel Intensity:** $0.1412$
- **Generated Mean Pixel Intensity:** $0.1212$
- **Real Background Sparsity ($< 0.1$ intensity):** $81.7\%$
- **Generated Background Sparsity ($< 0.1$ intensity):** $74.4\%$
- **End-to-End Reconstruction MSE:** $0.0463$

### 7.3 Independent Classifier Recognizability Assessment
Passing the DBN-synthesized digits through the independently trained MNIST CNN classifier yielded:
- **Recognizable Ratio ($\text{Confidence} \ge 70\%$):** $44.0\%$ ($11/25$ samples)
- **Average Classifier Confidence:** $65.3\%$
- **Synthesized Digit Distribution:** Samples produced digits across multiple classes: 0, 1, 2, 3, 4, 5, 7, 8.

---

## 8. SYSTEM TESTING & VERIFICATION

Automated testing was conducted using `pytest 9.1.1` on Python 3.12:
- `test_rbm.py`: Verified parameter shapes, probability bounds $[0, 1]$, binary Bernoulli sampling, finite free energy, and parameter updates.
- `test_dbn.py`: Verified multi-layer stacking, upward propagation, downward generation, roundtrip reconstruction, and checkpoint persistence.
- `test_generation.py`: Verified output dimensions $(N, 28, 28)$, value range $[0.0, 1.0]$, and deterministic reproducibility under fixed random seeds.

**Test Execution Outcome:**
`13 passed in 11.83s (100% test pass rate)`

---

## 9. CONCLUSION & FUTURE WORK

### 9.1 Conclusion
This project successfully designed, implemented, and validated a Deep Belief Network for handwritten digit generation. By adhering strictly to foundational energy-based graphical formulations, the implementation demonstrates how stacked RBMs learn hierarchical representations and how alternating Gibbs sampling in top associative memory enables ancestral image synthesis. The software package meets all college mini-project standards, operating cleanly on everyday laptop hardware with full academic rigor.

### 9.2 Future Scope
1. **Persistent Contrastive Divergence (PCD):** Implementing continuous fantasy particle chains across mini-batch updates to accelerate Markov chain convergence.
2. **Gaussian-Bernoulli Units:** Extending the visible layer from binary pixels to Gaussian continuous distributions to model complex natural image datasets (e.g., CIFAR-10, SVHN).
3. **Class-Conditioned DBNs:** Augmenting the top associative memory with one-hot class label clamping to allow deterministic user steering of generated digits.
4. **End-to-End Wake-Sleep Fine-Tuning:** Applying the full Helmholtz machine wake-sleep algorithm to decouple and optimize recognition and generative weights.

---

## 10. REFERENCES
1. Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets. *Neural Computation*, 18(7), 1527-1554.
2. Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 1771-1800.
3. LeCun, Y., Cortes, C., & Burges, C. J. (2010). MNIST handwritten digit database. *AT&T Labs*.
4. Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. *Proceedings of the 25th International Conference on Machine Learning (ICML)*, 1064-1071.
5. Salakhutdinov, R., & Hinton, G. (2009). Deep Boltzmann machines. *Artificial Intelligence and Statistics (AISTATS)*, 448-455.
