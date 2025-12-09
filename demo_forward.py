"""
Simple demo script for the Transformer encoder.

Runs a single forward pass on random input and prints tensor shapes.
"""

import torch
from transformer_encoder import TransformerEncoderBlock  # make sure this matches your file/class name


def main():
    # These must match the encoder configuration in transformer_encoder.py
    d_model = 64
    num_heads = 4
    d_hidden = 128
    seq_len = 10
    batch_size = 2

    # Instantiate the encoder block
    encoder = TransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_hidden=d_hidden,
    )

    # Dummy input: [batch_size, seq_len, d_model]
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    out = encoder(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
