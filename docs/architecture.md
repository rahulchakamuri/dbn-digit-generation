# Deep Belief Network Architecture & Mathematical Derivations

## 1. Architectural Overview

A **Deep Belief Network (DBN)** is a hybrid probabilistic graphical model consisting of multiple stacked layers of latent stochastic variables.
Introduced by Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh (2006), the architecture consists of:
1. An **undirected bipartite associative memory** in the top two layers.
2. A **directed acyclic belief network** with top-down generative connections in all lower layers.

```
                    +-----------------------------+
                    |    Hidden Units (h3)        |  (100 units)
                    |     Layer 3 Representation   |
                    +--------------+--------------+
                                   ^
                 Top Undirected    |  Symmetric Weights W3, Biases a3, b3
                 Associative RBM   v
                    +--------------+--------------+
                    |    Hidden Units (h2)        |  (250 units)
                    |     Layer 2 Representation   |
                    +--------------+--------------+
                                   |
                  Directed         |  Downward Generative Weights W2^T, a2
                  Belief Net       v
                    +--------------+--------------+
                    |    Hidden Units (h1)        |  (500 units)
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

## 2. Restricted Boltzmann Machine (RBM) Formulation

### 2.1 Graphical Structure
An RBM is an energy-based undirected bipartite graph.
- Visible vector: $\mathbf{v} = (v_1, v_2, \dots, v_D)^T \in \{0, 1\}^D$
- Hidden vector: $\mathbf{h} = (h_1, h_2, \dots, h_K)^T \in \{0, 1\}^K$
- Trainable parameters $\theta = \{\mathbf{W}, \mathbf{a}, \mathbf{b}\}$:
  - $\mathbf{W} \in \mathbb{R}^{D \times K}$: bipartite weight matrix
  - $\mathbf{a} \in \mathbb{R}^D$: visible bias vector
  - $\mathbf{b} \in \mathbb{R}^K$: hidden bias vector

### 2.2 Energy Function
The energy of a joint configuration $(\mathbf{v}, \mathbf{h})$ is:
$$E(\mathbf{v}, \mathbf{h}; \theta) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h} = -\sum_{i=1}^D a_i v_i - \sum_{j=1}^K b_j h_j - \sum_{i=1}^D \sum_{j=1}^K v_i W_{ij} h_j$$

### 2.3 Joint Probability and Partition Function
The joint distribution is defined by the Boltzmann-Gibbs distribution:
$$P(\mathbf{v}, \mathbf{h}; \theta) = \frac{e^{-E(\mathbf{v}, \mathbf{h}; \theta)}}{Z(\theta)}$$
where $Z(\theta)$ is the intractable normalization constant (partition function):
$$Z(\theta) = \sum_{\mathbf{v} \in \{0, 1\}^D} \sum_{\mathbf{h} \in \{0, 1\}^K} e^{-E(\mathbf{v}, \mathbf{h}; \theta)}$$

### 2.4 Factorized Conditional Distributions
Because there are strictly no connections within layers:
$$P(\mathbf{h} \mid \mathbf{v}) = \prod_{j=1}^K P(h_j \mid \mathbf{v}), \quad P(\mathbf{v} \mid \mathbf{h}) = \prod_{i=1}^D P(v_i \mid \mathbf{h})$$
With Bernoulli units:
$$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_{i=1}^D v_i W_{ij}\right) = \frac{1}{1 + \exp\left(-\left(b_j + \sum_i v_i W_{ij}\right)\right)}$$
$$P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_{j=1}^K W_{ij} h_j\right) = \frac{1}{1 + \exp\left(-\left(a_i + \sum_j W_{ij} h_j\right)\right)}$$

---

## 3. Contrastive Divergence (CD-k) Training Derivation

### 3.1 Log-Likelihood Gradient
The marginal probability of a visible vector $\mathbf{v}$ is:
$$P(\mathbf{v}) = \frac{\sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}}{Z} = \frac{e^{-F(\mathbf{v})}}{Z}$$
where $F(\mathbf{v})$ is the **Free Energy**:
$$F(\mathbf{v}) = -\ln \sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})} = -\mathbf{a}^T \mathbf{v} - \sum_{j=1}^K \ln\left(1 + e^{\mathbf{v}^T \mathbf{W}_{*j} + b_j}\right) = -\mathbf{a}^T \mathbf{v} - \sum_{j=1}^K \text{softplus}\left(\mathbf{v}^T \mathbf{W}_{*j} + b_j\right)$$

Differentiating the log-likelihood of training data:
$$\frac{\partial \ln P(\mathbf{v})}{\partial W_{ij}} = -\frac{\partial F(\mathbf{v})}{\partial W_{ij}} - \frac{\partial \ln Z}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

### 3.2 Contrastive Divergence Approximation
Hinton's Contrastive Divergence replaces the intractable model expectation $\langle \cdot \rangle_{\text{model}}$ by running a $k$-step Markov chain initialized at the data sample $\mathbf{v}^{(0)}$:
1. Sample $\mathbf{h}^{(0)} \sim P(\mathbf{h} \mid \mathbf{v}^{(0)})$.
2. For $t = 1, \dots, k$:
   - Sample $\mathbf{v}^{(t)} \sim P(\mathbf{v} \mid \mathbf{h}^{(t-1)})$.
   - Sample $\mathbf{h}^{(t)} \sim P(\mathbf{h} \mid \mathbf{v}^{(t)})$.
3. Update parameters with learning rate $\eta$, momentum $\gamma$, and weight decay $\lambda$:
   $$\Delta W_{ij} = \gamma \Delta W_{ij}^{(\text{prev})} + \eta \left( v_i^{(0)} P(h_j^{(0)}=1 \mid \mathbf{v}^{(0)}) - v_i^{(k)} P(h_j^{(k)}=1 \mid \mathbf{v}^{(k)}) \right) - \lambda W_{ij}$$
   $$\Delta a_i = \gamma \Delta a_i^{(\text{prev})} + \eta \left( v_i^{(0)} - v_i^{(k)} \right)$$
   $$\Delta b_j = \gamma \Delta b_j^{(\text{prev})} + \eta \left( P(h_j^{(0)}=1 \mid \mathbf{v}^{(0)}) - P(h_j^{(k)}=1 \mid \mathbf{v}^{(k)}) \right)$$

---

## 4. Probabilistically Justified Generative Sampling Strategy

In a DBN, generation proceeds hierarchically:

### Phase 1: MCMC Mixing in Top Associative Memory
The top layer $\text{RBM}_3$ is an undirected model. We initialize its visible units $\mathbf{h}_2^{(0)}$ with uniform Bernoulli bits. We run alternating Gibbs sampling for $M$ steps ($M \ge 100$):
$$\mathbf{h}_3^{(t)} \sim P(\mathbf{h}_3 \mid \mathbf{h}_2^{(t-1)}; \mathbf{W}_3, \mathbf{b}_3)$$
$$\mathbf{h}_2^{(t)} \sim P(\mathbf{h}_2 \mid \mathbf{h}_3^{(t)}; \mathbf{W}_3^T, \mathbf{a}_3)$$
At convergence ($t = M$), $(\mathbf{h}_2^{(M)}, \mathbf{h}_3^{(M)})$ represents a sample from the joint prior $P(\mathbf{h}_2, \mathbf{h}_3)$.

### Phase 2: Directed Ancestral Downward Propagation
Using $\mathbf{h}_2^{(M)}$ from the top equilibrium state, we project activations downwards through the directed sigmoid belief layers:
1. **Pass through Layer 2:**
   $$P(\mathbf{h}_1 = 1 \mid \mathbf{h}_2) = \sigma\left(\mathbf{a}_2 + \mathbf{h}_2 \mathbf{W}_2^T\right)$$
2. **Pass through Layer 1:**
   $$P(\mathbf{v}_0 = 1 \mid \mathbf{h}_1) = \sigma\left(\mathbf{a}_1 + \mathbf{h}_1 \mathbf{W}_1^T\right)$$
3. **Continuous Grayscale Reconstruction:**
   $\mathbf{v}_0 \in [0.0, 1.0]^{784}$ is reshaped to $(28, 28)$ to yield the synthetic digit image.
