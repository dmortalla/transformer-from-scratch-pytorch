# Transformer Encoder From Scratch (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Model](https://img.shields.io/badge/Architecture-Transformer-purple)
![Educational](https://img.shields.io/badge/Purpose-Educational-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A minimal, readable Transformer encoder implemented in PyTorch, designed to emphasize architecture clarity. Demonstrates LayerNorm ordering, attention mechanisms, residual connections, and feedforward blocks in a pedagogical format.

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

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

