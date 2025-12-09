# Transformer Encoder From Scratch — PyTorch Implementation

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Model](https://img.shields.io/badge/Architecture-Transformer-purple)
![Educational](https://img.shields.io/badge/Purpose-Educational-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## ⚡ Tagline
A clean, minimal PyTorch implementation of a Pre-LN Transformer encoder block built entirely from first principles.

---

## 🚀 Quickstart Demo (For Reviewers)

Run a **lightweight forward-pass demo**:

```bash
pip install -r requirements.txt
python demo_forward.py
```

This validates the model architecture, attention mechanism, and end-to-end forward computation using synthetic data.

---

## 📦 Full Model Run

Run the full Transformer encoder implementation:

```bash
python transformer_encoder.py
```

This executes the complete encoder block with attention, MLP layers, residual connections, and normalization.

---

## 📁 Files

```text
transformer_encoder.py   # Full Transformer encoder implementation
demo_forward.py          # Lightweight forward-pass demonstration
requirements.txt         # Dependencies
```

---

## 🏗 Overview

This implementation follows the standard **Pre-LayerNorm Transformer Encoder** pattern:

```
Input → LayerNorm → Multi-Head Attention → Residual →
LayerNorm → MLP (GELU) → Residual → Output
```

It includes:

- Scaled dot-product multi-head self-attention  
- Projection layers for Q/K/V  
- Attention output projection  
- Position-wise MLP with GELU  
- Residual skip connections  
- LayerNorm for training stability  

This repo is ideal for showcasing architectural understanding of modern Transformer building blocks.

---

## 📂 Project Structure

```text
.
├── transformer_encoder.py
├── demo_forward.py
├── requirements.txt
├── CONTRIBUTING.md
└── SECURITY.md
```

---

## 🧱 Architecture Overview

The encoder block follows the common **Pre-LN Transformer** pattern:

```bash
Input embeddings
        |
        v
   LayerNorm (LN1)
        |
        v
 Multi-Head Self-Attention
        |
        v
 Residual Add ---------------+
        |                    |
        v                    |
   LayerNorm (LN2)           |
        |                    |
        v                    |
 Position-wise Feedforward   |
 (MLP / GELU)                |
        |                    |
        v                    |
 Residual Add <--------------+
        |
        v
  Encoder output
```

---

## 🤝 Contributing
See `CONTRIBUTING.md` for contribution workflow and coding standards.

---

## 📄 License
MIT License. See `LICENSE` for details.
