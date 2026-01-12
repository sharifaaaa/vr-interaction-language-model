# vr_transformer/__init__.py
"""
Public API for the vr_transformer package.

Re-exports:
- runner utilities: run_attention, get_chunk_batch
- builders and runners from attention_utils
- core modules: MultiHeadAttention, TransformerEncoder, TransformerEncoderLayer
"""

from .attention_runner import (
    run_attention,
    get_chunk_batch,
)

from .attention_utils import (
    build_single_head,
    build_multi_head,
    build_encoder,
    run_single_head_chunk,
    run_multi_head_chunk,
    run_multi_head_batch,
    run_encoder_chunk,
    run_encoder_batch,
)

from .multihead_attention import (
    MultiHeadAttention,
    RotaryEmbedding,
    sinusoidal_pe,
    init_ as init_weights,   # optional: convenient alias
)

from .transformer_blocks import (
    TransformerEncoderLayer,
    TransformerEncoder,
)

from .transformer_classifier import TransformerClassifier
from single_attention import SingleHeadAttention


__all__ = [
    # attention_runner
    "run_attention", "get_chunk_batch",
    # attention_utils
    "build_single_head", "build_multi_head", "build_encoder",
    "run_single_head_chunk", "run_multi_head_chunk", "run_multi_head_batch",
    "run_encoder_chunk", "run_encoder_batch",
    # multihead_attention
    "MultiHeadAttention", "RotaryEmbedding", "sinusoidal_pe", "init_weights",
    # transformer_blocks
    "TransformerEncoderLayer", "TransformerEncoder",
    #transformer_classifier
    "TransformerClassifier"

]

__version__ = "0.1.0"
