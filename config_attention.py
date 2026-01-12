import torch
# --------- config of Vr Transformer(set these before running) -----------------
"""
    Multi-head scaled dot-product attention with optional positional embeddings.
    Supports inputs of shape:
      - (L, E)             -> treated as batch size 1
      - (B, L, E)

    Args
    ----
    d_model : token embedding size (E)
    n_heads : number of heads (h)
    d_k     : key/query size per head
    d_v     : value size per head (defaults to d_k)
    pos_type: "learned" | "sinusoidal" | "none"
    max_len : maximum sequence length for positions
    init    : initialization for projection weights: "xavier_uniform"|"xavier_normal"|"kaiming_uniform"|"kaiming_normal"|"zeros"
    seed    : manual seed for reproducibility

    Projections are stored as nn.Parameter to allow manual setting.
    Shapes:
      WQ, WK : (d_model, n_heads * d_k)
      WV     : (d_model, n_heads * d_v)
      WO     : (n_heads * d_v, d_model)
    """
ATTN_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "d_k": 32,   #must be even when pos-type = "RoPE"
    "d_v": None,
    "pos_type": "rope",         # "learned", "sinusoidal", "none", "rope"
    "max_len": 240,   #chunk_size
    "stride" : 120,    #Chunk within each contiguous run. Optional overlap with `stride`.
    "init": "xavier_uniform",
    "seed": 42,

    # encoder extras:
    "num_layers": 3,     # stack depth
    "ffn_mult": 4,       # or it is the same mlp_ratio .FFN width = ffn_mult * d_model or FFN width = mlp_ratio * d_model
    "dropout": 0.15,
    "layer_norm_eps": 1e-5,

}
POOLING = "cls"   # "mean" | "max" | "cls"   # --- pooling for window embedding ---
MODE = "evaluate"      # "single", "multi", "batch", "enc-single", "enc-batch",
                                #or "clf-batch" or "clf-single" or "train-clf" (for classifier)
                                #"pretrain" or "finetune" or "interpret" or "evaluate" or "fusion_with_audio"
                                #"embed-viz"


USE_PRETRAINED = True     # if True, finetune must use pretrained encoder and load PRETRAIN_CKPT_PATH (error if missing)

SAVE_BASELINE = True   # or False if you don’t want the before-training baseline

# --- debugging / printing ---
PRINT_POOLED: bool = False
MAX_PRINT: int = 5

