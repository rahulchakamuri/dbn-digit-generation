# Viva Voce & Technical Review Preparation Guide

## Project Title
**Deep Belief Network for Synthetic Handwritten Digit Generation Using Stacked Restricted Boltzmann Machines**

---

## 1. Two-Minute Project Elevator Pitch

> *"Good morning, respected members of the evaluation committee.*
>
> *Our project is titled **'Deep Belief Network for Synthetic Handwritten Digit Generation Using Stacked Restricted Boltzmann Machines'**.*
>
> *The core objective is to investigate how deep probabilistic energy-based models learn complex multimodal image distributions and synthesize novel handwritten digits from scratch without relying on modern continuous GAN or VAE architectures.*
>
> *To accomplish this, we constructed a **Deep Belief Network (DBN)** composed of three stacked **Restricted Boltzmann Machines (RBMs)** using PyTorch. We utilized **greedy layer-wise pretraining**, where each RBM is trained sequentially on the hidden activations of the layer beneath it using **Contrastive Divergence ($CD_1$)**.*
>
> *For synthetic digit generation, we implemented a mathematically rigorous two-phase procedure:*
> *First, we run **alternating Gibbs sampling** in the top undirected RBM layer for 100 steps to reach equilibrium on the learned joint prior.*
> *Second, we perform an **ancestral downward pass** through the lower directed belief layers, projecting the latent features into visible $28 \times 28$ pixel intensities.*
>
> *We evaluated the network using end-to-end reconstruction error, pixel intensity distribution histograms, and an independent MNIST CNN classifier to verify stroke recognizability.*
>
> *Finally, we packaged the entire system into an interactive **Streamlit web application** that runs seamlessly on standard laptop CPUs. Thank you."*

---

## 2. Five-Minute Viva Demonstration Script

1. **Step 1: Introduction & System Health (0:00 - 1:00)**
   - Start Streamlit app: `streamlit run app.py`.
   - Point to the **Sidebar**: show that CPU execution is active and pre-trained checkpoint `dbn_mnist_demo.pt` is recognized.
   - Explain the block architecture diagram on the **Home Dashboard**.

2. **Step 2: Dataset Exploration (1:00 - 2:00)**
   - Open **`1_Dataset_Explorer`**.
   - Show the 60,000-image dataset details ($28 \times 28$ resolution, 784 flattened dimensions, $[0, 1]$ normalized).
   - Display the pixel histogram: explain that ~80% of pixels are zeros (background) and explain why Bernoulli RBMs are well-suited for this distribution.

3. **Step 3: Training Diagnostics (2:00 - 3:00)**
   - Open **`2_Model_Training`**.
   - Show the pretraining loss and reconstruction MSE curves for RBM 1, RBM 2, and RBM 3.
   - Explain that each layer compressed features into progressively higher-level abstractions.

4. **Step 4: Live Digit Generation (3:00 - 4:00)**
   - Open **`3_Generate_Digits`**.
   - Select 25 samples and 100 Gibbs steps. Click **Generate Synthetic Digits**.
   - Explain the top-level Gibbs sampling and downward synthesis.
   - Demonstrate the download buttons for PNG collage and ZIP archive.

5. **Step 5: Evaluation & Viva Defense (4:00 - 5:00)**
   - Open **`4_Real_vs_Generated`** and **`5_Evaluation`**.
   - Show the side-by-side comparison, highlighting the structural similarities and characteristic soft edges.
   - Present the independent CNN recognizability metrics and state the academic disclaimer.

---

## 3. Comprehensive Viva Voce Questions & Answers

### Category A: RBM Foundations & Mathematics

#### Q1: What is a Restricted Boltzmann Machine (RBM)?
**Answer:** An RBM is an energy-based, undirected bipartite graphical neural network consisting of a visible layer $\mathbf{v}$ and a hidden layer $\mathbf{h}$. It is termed "restricted" because connections exist solely between visible and hidden units; there are strictly no visible-to-visible or hidden-to-hidden connections.

#### Q2: What is the Energy Function of an RBM?
**Answer:** The energy of a joint visible-hidden configuration is defined as:
$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h} = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i W_{ij} h_j$$
where $\mathbf{W}$ is the weight matrix, $\mathbf{a}$ is the visible bias vector, and $\mathbf{b}$ is the hidden bias vector. Configurations with lower energy have higher probability under the Boltzmann distribution.

#### Q3: What is the joint probability distribution and why is the partition function intractable?
**Answer:** The joint probability is $P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{Z}$, where $Z = \sum_{\mathbf{v}} \sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$ is the partition function. Computing $Z$ is NP-hard for large networks because it requires summing over $2^{D+K}$ discrete states (e.g. $2^{784 + 500} \approx 2^{1284}$), which is computationally impossible.

#### Q4: Why is the bipartite restriction so important mathematically?
**Answer:** Because visible units have no lateral connections, they are conditionally independent given the hidden units: $P(\mathbf{v} \mid \mathbf{h}) = \prod_i P(v_i \mid \mathbf{h})$. Similarly, hidden units are conditionally independent given the visible units: $P(\mathbf{h} \mid \mathbf{v}) = \prod_j P(h_j \mid \mathbf{v})$. This allows exact, parallel calculation of conditional probabilities:
$$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_i v_i W_{ij}\right), \quad P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_j W_{ij} h_j\right)$$

#### Q5: What is Free Energy in an RBM?
**Answer:** Free energy $F(\mathbf{v})$ is the effective energy assigned to a visible vector after analytically marginalizing out all hidden unit configurations:
$$e^{-F(\mathbf{v})} = \sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})} \implies F(\mathbf{v}) = -\mathbf{a}^T \mathbf{v} - \sum_{j} \text{softplus}(\mathbf{v} \mathbf{W}_{*j} + b_j)$$
This formulation allows log-likelihood gradients to be calculated directly.

---

### Category B: Contrastive Divergence & Pretraining

#### Q6: What is Contrastive Divergence (CD-k)?
**Answer:** Contrastive Divergence is an approximation algorithm introduced by Geoffrey Hinton to train RBMs without running an infinite MCMC chain. The exact gradient of the log-likelihood is $\frac{\partial \ln P(\mathbf{v})}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$. CD-$k$ approximates the second expectation by running Gibbs sampling for only $k$ steps (typically $k=1$) initialized at the data vector $\mathbf{v}^{(0)}$.

#### Q7: What are the Positive and Negative phases in CD?
**Answer:**
- **Positive Phase:** Driven by real training data $\mathbf{v}^{(0)}$. We calculate the hidden unit probabilities $P(\mathbf{h}^{(0)}=1 \mid \mathbf{v}^{(0)})$ and form the empirical data correlation matrix $\mathbf{v}^{(0)T} \mathbf{h}^{(0)}$.
- **Negative Phase:** The network reconstructs visible units $\mathbf{v}^{(k)}$ and hidden units $\mathbf{h}^{(k)}$ through $k$ Gibbs sampling steps, computing the model correlation matrix $\mathbf{v}^{(k)T} \mathbf{h}^{(k)}$.
- **Update Rule:** $\Delta W_{ij} \propto \langle v_i^{(0)} h_j^{(0)} \rangle - \langle v_i^{(k)} h_j^{(k)} \rangle$. The update lowers the energy of real data configurations and raises the energy of hallucinated model configurations.

#### Q8: Why do we use probabilities instead of binary samples in the negative phase?
**Answer:** While stochastic binary sampling is required for intermediate Gibbs Markov chain transitions, using continuous probabilities $P(h_k = 1 \mid v_k)$ for the final hidden update significantly reduces sampling variance and accelerates training convergence.

#### Q9: What is Greedy Layer-Wise Pretraining?
**Answer:** Introduced by Hinton et al. in 2006, greedy layer-wise pretraining trains a deep multi-layer network one layer at a time without backpropagation. First, RBM 1 is trained on raw pixel vectors. Then, RBM 1's hidden unit activations are frozen and used as training inputs for RBM 2. Then RBM 2's activations train RBM 3. Mathematically, adding each layer is guaranteed to improve a variational lower bound on the training data log-likelihood.

#### Q10: Why did layer-wise pretraining solve the vanishing gradient problem?
**Answer:** Prior to 2006, end-to-end backpropagation failed in deep networks because gradients vanished exponentially when backpropagated through multiple sigmoid layers. Greedy layer-wise pretraining places each layer's weights in a well-conditioned basin of attraction where feature detectors are already meaningful, bypassing gradient decay.

---

### Category C: DBN Architecture & Generative Sampling

#### Q11: What is the exact architectural difference between an RBM and a DBN?
**Answer:** An RBM is a single-layer bipartite model with bidirectional, undirected connections. A DBN is a multi-layer stack of RBMs where the top two layers remain an undirected associative memory, while all lower layers act as a directed acyclic belief network with downward generative connections $\mathbf{W}^T$.

#### Q12: How does image generation work in a DBN?
**Answer:** Generation is a mathematically justified two-step procedure:
1. **Top-Level MCMC Mixing:** The top associative RBM is initialized with random Bernoulli states and run through $M$ alternating Gibbs sampling steps ($M \ge 100$) until the Markov chain reaches stationary equilibrium on the joint prior $P(\mathbf{h}_2, \mathbf{h}_3)$.
2. **Downward Directed Ancestral Pass:** The equilibrium state $\mathbf{h}_2$ is propagated downwards through the directed belief network: $\mathbf{h}_2 \to \mathbf{h}_1 \to \mathbf{v}_0$ via $P(\mathbf{h}_{l-1} = 1 \mid \mathbf{h}_l) = \sigma(\mathbf{h}_l \mathbf{W}_l^T + \mathbf{a}_l)$. The final output $\mathbf{v}_0 \in [0, 1]^{784}$ is reshaped to $(28, 28)$ to form the synthetic digit image.

#### Q13: What is Gibbs Sampling and why is it used?
**Answer:** Gibbs sampling is a Markov Chain Monte Carlo (MCMC) algorithm for generating observations from a complex joint multivariate distribution. Because computing the joint distribution $P(\mathbf{v}, \mathbf{h})$ directly is intractable, Gibbs sampling updates each variable conditioned on the current values of all other variables. In an RBM, all hidden units can be updated in parallel given visible units, and vice versa.

#### Q14: What is Burn-In in Gibbs Sampling?
**Answer:** Burn-in refers to the initial iterations of a Markov chain that are discarded to ensure that the chain has traversed away from arbitrary random starting states and settled into its stationary equilibrium distribution. In our project, 100 Gibbs steps ensure adequate burn-in.

#### Q15: Why do some generated digits resemble blends of two digits?
**Answer:** In an energy-based model, each digit class represents an attractor basin (energy well). As the Gibbs sampler traverses the energy landscape, it can wander near saddle points between two basins (e.g. between the manifold of digit 3 and digit 5), sampling chimeric or ambiguous feature combinations.

---

### Category D: Evaluation, Comparisons, and Engineering

#### Q16: How does a DBN compare with a GAN?
**Answer:**
- **DBN:** An energy-based probabilistic model trained via layer-wise Contrastive Divergence; samples via top-level MCMC followed by downward synthesis; highly stable training; runs easily on CPU.
- **GAN:** A game-theoretic min-max optimization between Generator and Discriminator; generates samples in a single forward pass; produces sharper images but suffers from mode collapse and training instability; requires GPU.

#### Q17: How does a DBN compare with a VAE?
**Answer:**
- **DBN:** Uses unconstrained binary latent states; trained layer-by-layer; top layer is an undirected associative memory; samples via MCMC.
- **VAE:** Uses continuous Gaussian latent spaces; trained end-to-end via the Evidence Lower Bound (ELBO) with the reparameterization trick; samples via a single pass from an isotropic Gaussian prior.

#### Q18: What is the purpose of the independent CNN classifier in this project?
**Answer:** It provides an objective, automated proxy metric to assess whether the synthetic strokes match recognizable digit features. It was trained independently on MNIST and was completely isolated from DBN training.

#### Q19: Why can classifier confidence NOT be equated with human-perceived realism?
**Answer:** Neural network classifiers are sensitive to local stroke correlations and spatial frequencies. A classifier can assign 95% confidence to a noisy or slightly deformed numeral if key discriminating features are present, whereas a human observer might easily detect unnatural artifacts or fuzzy halos.

#### Q20: What preprocessing was applied to MNIST and why?
**Answer:** MNIST pixel values were normalized from $[0, 255]$ integers to $[0.0, 1.0]$ floats and flattened from $(28, 28)$ into 784-dimensional vectors. In a Bernoulli RBM, pixel values in $[0, 1]$ represent the activation probability $P(v_i = 1)$, which natively matches the output of the sigmoid activation function.

#### Q21: What are the primary failure modes of RBM training?
**Answer:**
1. **Diverging Weights:** If learning rate is too high, weights grow excessively large, causing sigmoid activations to saturate at 0 or 1.
2. **Under-mixing:** If CD steps $k$ are too small or Gibbs burn-in is insufficient, the model only learns local reconstructions rather than the global distribution.
3. **Dead Units:** Hidden units with strong negative biases that never activate.

#### Q22: What hyperparameters were chosen for this implementation?
**Answer:**
- **Learning Rate:** $\eta = 0.05$ (stable SGD updates)
- **Momentum:** $\gamma = 0.5$ (accelerates convergence across flat plateaus)
- **Weight Decay:** $\lambda = 0.0001$ ($L_2$ regularization to prevent weight explosion)
- **Batch Size:** 64
- **Initial Weights:** $\mathcal{N}(0, 0.01^2)$ Gaussian distribution to break symmetry.

#### Q23: Why does our project support both Demo Mode and Full Training Mode?
**Answer:** Academic viva demonstrations must run reliably on everyday laptops without dedicated GPUs. Demo Mode ($[784, 256, 128, 64]$, 6,000 samples, 3-5 epochs) executes in under 15 seconds, allowing full interactive testing during presentations. Full Training Mode ($[784, 500, 250, 100]$) provides full capacity for deep feature hierarchies.

#### Q24: What is Persistent Contrastive Divergence (PCD)?
**Answer:** In standard CD, the Gibbs chain is restarted at the data vector $\mathbf{v}^{(0)}$ for every mini-batch. In PCD (Tieleman, 2008), persistent fantasy particles maintain continuous Markov chains across mini-batch updates, providing a better approximation of the model's true stationary distribution.

#### Q25: What future enhancements could be added to this project?
**Answer:**
1. **Persistent Contrastive Divergence (PCD)** for better MCMC mixing.
2. **Gaussian-Bernoulli RBMs** for continuous real-world images like CIFAR-10.
3. **Class-Conditional DBNs** by clamping a one-hot label vector to the top associative RBM to generate specific user-selected digits.
4. **Wake-Sleep Fine-Tuning** to optimize generative and recognition weights end-to-end.
