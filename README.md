# Transformer Encoder From Scratch — PyTorch Implementation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## ⚡ Tagline
A clean, minimal PyTorch implementation of a Pre-LN Transformer encoder block built entirely from first principles.

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python transformer_encoder.py
```

Runs a full forward pass through the encoder using synthetic inputs.

---

## 📁 Files

```text
transformer_encoder.py   # Full transformer implementation
run_demo.py              # Lightweight forward-pass demo
requirements.txt         # Dependencies
```

---

## 🏗 Overview

This implementation follows the standard **Pre-LayerNorm Transformer Encoder** pattern:

```
Input → LayerNorm → Multi-Head Attention → Residual →
LayerNorm → MLP (GELU) → Residual → Output
```

Includes:

- Scaled dot-product self-attention  
- Multi-head attention mechanism  
- Position-wise feed-forward network  
- GELU activations  
- Residual skip connections  
- LayerNorm for stability  

---

## 📂 Project Structure

```text
.
├── transformer_encoder.py
├── run_demo.py
├── requirements.txt
└── CONTRIBUTING.md
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
See CONTRIBUTING.md for contribution workflow and code style guidelines.

---

## 📄 License
MIT License. See `LICENSE` for details.
