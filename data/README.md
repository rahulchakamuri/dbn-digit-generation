# MNIST Dataset Directory

This directory stores the MNIST handwritten digit dataset downloaded via PyTorch torchvision utilities:

- **Source**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges (Yann LeCun's MNIST database)
- **Training Samples**: 60,000 grayscale images ($28 \times 28$ pixels)
- **Test Samples**: 10,000 grayscale images ($28 \times 28$ pixels)
- **Labels**: Digits 0 through 9
- **Format**: Byte tensors normalized to $[0.0, 1.0]$ and flattened to $784$ dimensions for Bernoulli RBM inputs.

> **Note**: Raw dataset archives (`MNIST/raw/*`) are ignored by `.gitignore` so they are not committed to source control. They will be downloaded automatically when running `models/train.py` or launching the Streamlit app.
