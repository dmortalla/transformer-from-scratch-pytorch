# Transformer Encoder From Scratch (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![Transformer](https://img.shields.io/badge/Architecture-Transformer-blueviolet)
![Attention](https://img.shields.io/badge/Module-Attention-ff69b4)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

> Minimal Transformer encoder implemented from scratch, including multi-head self-attention, QKV projections, residual connections, LayerNorm, and sinusoidal positional encoding.

---

## 🚀 Overview

This project implements a **Transformer encoder block entirely from scratch** in PyTorch — no `nn.Transformer`, no shortcuts.  
It mirrors the internal building blocks found in modern LLMs.

Included components:

- Scaled dot-product attention  
- Multi-head projection  
- Pre-LayerNorm style normalization  
- Residual connections  
- Feedforward MLP block  
- Sinusoidal positional encodings  

Perfect for studying “attention under the hood” and debugging LLM internals.

---

## ▶️ Quickstart

```bash
pip install -r requirements.txt
python transformer_encoder.py

transformer_encoder.py    # Full encoder implementation
requirements.txt
```

---

## 🧱 Architecture Overview

The encoder block follows the common **Pre-LN Transformer** pattern:

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

---

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

