# config_pretrain.py
FINETUNE_BACKBONE = "VRtransformer"  # "temporal" or "VRtransformer"
PRETRAIN_CKPT_PATH = "encoder_pretrained.pt" # It adds _vr when Backbone = Vrtransformer
# --- objectives switches ---
OBJECTIVES = {
    "mtm": True,          # Masked Time-Series Modeling (MAE/BERT-style)
    "contrastive": True,  # InfoNCE on augmented views
    "xmodal": False,       # behavior ↔ text alignment when speech exists
}

# --- masking (MTM) ---
MASK_TOKEN_SCHEME = {
    "token_p":   0.15,   # % of timesteps to mask
    "span_frac": 0.50,   # half of masks are short spans (2–12 steps)
    "var_p":     0.10,   # probability to mask whole variable groups
    "min_span":  2,
    "max_span":  12,
}

# --- light augmentations (contrastive) ---
AUGS = {
    "temporal_crop_min": 0.90,  # keep 90–100% of the window
    "temporal_crop_max": 1.00,
    "time_jitter_steps": 1,     # ±1 step
    "feat_dropout_p":    0.10,
    "gauss_sigma":       0.02,  # after z-norm
}

# --- losses / temps ---
LAMBDA = {
    "mtm": 1.0,
    "contrastive": 1.0,
    "xmodal": 0.5,
}
TEMPERATURE = 0.30  #0.10  # Hyper parameter to tune the Contrastive Loss - The InfoNCE temperature (TEMPERATURE = 0.10) controls the sharpness of the
# distribution. A very low temperature can lead to very sharp gradients and unstable training.

# --- training/optimization ---
PRETRAIN_EPOCHS = 45  #50
FINETUNE_EPOCHS = 15   #30
#UNFREEZE_MODE = "abrupt"   # "abrupt" or "gradual"
#LAYERS_PER_UNFREEZE = 1     # how many encoder layers to release each epoch
FREEZE_EPOCHS = 6           # keep encoder frozen for the first N epochs

LR_PRE  = 5e-5   #1e-4
LR   = 1e-5      #5e-5  # for Mode = 'trian_clf'
WEIGHT_DECAY = 1e-2
BATCH_SIZE_PRE = 32  #32
BATCH_SIZE  = 32   # 32
GRAD_CLIP_NORM = 1.0
SCHEDULER = "cosine"   # with warmup below
WARMUP_FRAC = 0.15     # ≈3 epochs warmup for 20-epoch fine-tune
EPOCHS: int = 30  #For mode = 'train-clf'
VAL_RATIO: float = 0.20   #It is used when we are not on DB split/stratified split
SEED: int = 42

# --- model head ---

DROPOUT_HEAD: float = 0.30   # 0.20–0.30 is the safe band; start at 0.25
HEAD_HIDDEN_RATIO = 1.0
LR_FT_HEAD = 3e-4             # head learns faster
LR_FT_ENC = 1e-5              # set to 0.0 while frozen; enable after epoch 5
"""
Quick guidance for HEAD_HIDDEN_RATIO:

    -Very small labeled set: HEAD_HIDDEN_RATIO = 0 (linear head) or 0.5.
    
    -Moderate data: 1.0 (hidden = d_model).
    
    -More data / underfitting: 1.5–2.0 (watch overfitting; keep dropout).
"""
#Extra parameter for pretrain Mode
VAL_FRACTION = 0.1
EARLY_STOP_PATIENCE = 5 #8     # 0 disables early stopping
EMA_ALPHA = 0.3             # 0.0 disables EMA; higher = faster response
PRETRAIN_PLOT_PATH = "./artifacts/pretrain/pretrain_temporal_loss.png"
PRETRAIN_PLOT_PATH_VR = "./artifacts/pretrain/pretrain_vr_loss.png"
STABILIZE_MA_WINDOW = 3   #length of Moving Average window for calculating smooth loss
STABILIZE_REL_TOL = 1e-3   # small tolerance - If the rate of change between the current and previous smoothed loss is less than some small tolerance, then loss has stabilized.

# --- checkpoints & mode flags ---

CLF_CKPT_PATH      = "emotion_classifier.pt"

METRICS = ("macro_f1", "per_class_f1", "confusion_matrix")

# --- loss selection for fine-tuning ---
LOSS_TYPE = "weighted_ce"      # "weighted_ce" or "focal"

# For weighted CE
USE_CLASS_WEIGHTS_CE = True    # if False, plain CE

# For focal loss
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_MODE = "class_weights"  # "none" | "vector" |"scalar" | "class_weights"
FOCAL_ALPHA_SCALAR = 0.25           # used only when MODE == "scalar"
FOCAL_ALPHA_NORMALIZE = True
"""
Meaning:

weighted_ce → CrossEntropy with class weights (if USE_CLASS_WEIGHTS_CE=True).

focal → Focal CE with γ=FOCAL_GAMMA; α(Alpha) taken from:

    -"none" → no α (Alpha)
    
    -"scalar" → α = FOCAL_ALPHA_SCALAR
    
    -"class_weights" → α = normalized class weights (good for imbalance).
    
    -For multiclass, a vector (from class_weights or user-provided) is usually better than a single scalar.
    
    -When LOSS_TYPE="focal", set USE_CLASS_WEIGHTS_CE=False 
        (which the code above does via use_ce_weights) to avoid double weighting.
"""


