# Contributing to Transformer From Scratch

Thank you for contributing to this educational, minimal Transformer implementation.  
This project prioritizes clarity, correctness, and approachability.

---

## 1. Fork the Repository

Click **Fork** (top-right on GitHub) to create your own copy of this repository.

---

## 2. Clone Your Fork & Create a Branch

```bash
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch
git checkout -b feature/your-feature-name
```

---

## 3. Make Your Changes

- Prioritize readability over complexity.
- Keep the encoder architecture aligned with standard **Pre-LN Transformer** design.
- Add tensor shape comments whenever helpful.
- Ensure compatibility with PyTorch 2.x.

---

## 4. Run Basic Checks

### Syntax Validation

```bash
python -m compileall .
```

### Forward Pass Sanity Check

```bash
python transformer_encoder.py
```

(Optional: Print shapes during the forward pass for validation.)

---

## 5. Open a Pull Request

- Explain what was changed and why.
- For architecture modifications, describe effects on:
  - tensor shapes  
  - computational cost  
  - performance  

---

## Code Style Guidelines

- Use clear, descriptive variable names.
- Add Google-style docstrings where appropriate.
- Avoid unnecessary abstraction—this project is meant for learning.
- Keep mathematical operations explicit and traceable.

---

## Thank You

Your contribution helps others learn Transformers from the ground up.

