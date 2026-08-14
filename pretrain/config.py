"""Model definition for Cognita-Mini-167M.

A 12-layer decoder-only transformer in the Qwen3 layout: pre-norm RMSNorm,
rotary position embeddings, SwiGLU feed-forward, grouped-query attention with
QK-norm, untied input and output embeddings. The weights are randomly
initialised here; nothing is loaded from a pre-trained Qwen checkpoint.

Run this file to print the parameter count:
    python pretrain/config.py
"""

from transformers import Qwen3Config, Qwen3ForCausalLM

VOCAB_SIZE = 32000
MAX_SEQ_LENGTH = 4096


def build_config(vocab_size: int = VOCAB_SIZE) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=896,
        num_hidden_layers=12,
        intermediate_size=2048,           # SwiGLU feed-forward width
        num_attention_heads=12,           # query heads, head_dim 128
        num_key_value_heads=4,            # grouped-query attention: 3 query heads share one KV head
        max_position_embeddings=MAX_SEQ_LENGTH,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def build_model(vocab_size: int = VOCAB_SIZE) -> Qwen3ForCausalLM:
    """A freshly initialised model, ready for pre-training."""
    return Qwen3ForCausalLM(build_config(vocab_size))


if __name__ == "__main__":
    model = build_model()
    print(f"Total parameters: {model.num_parameters():,} ({model.num_parameters() / 1e6:.2f}M)")
