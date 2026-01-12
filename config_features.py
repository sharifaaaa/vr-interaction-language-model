# config_features.py

HAND_QUALITY_COLS = ["has_valid_hands", "hands_imputed"]

BEHAVIORAL_COLS = [
    "gaze_x","gaze_y","gaze_z",
    "head_pos_x","head_pos_y","head_pos_z",
    "r_pos_x","r_pos_y","r_pos_z",
    "l_pos_x","l_pos_y","l_pos_z",
    "movement_speed","r_movement_speed","l_movement_speed",
]
#Imporatnt tip:gaze_actor, r_interacted_actor, l_interacted_actor are categorical
# → they must appear in CATEGORICAL_COLS. Adding them only to BEHAVIORAL_GROUPS doesn’t expose them to the tokenizer unless they’re also in CATEGORICAL_COLS.
CATEGORICAL_COLS = [
    "phase","area","speaker","gaze_actor",
    "r_interacted_actor","l_interacted_actor","session_density",
]   #Basic ones that collected by headSet

CATEGORICAL_COLS = CATEGORICAL_COLS + HAND_QUALITY_COLS

CAT_EMBED_DIM = 8    #Size of the vector used for each categorical column. Bigger = a bit more capacity, a bit slower.

# identifiers / bookkeeping that must survive merges & chunking,
# but are NEVER used as model inputs
ID_COLS = ["db_session_id", "split", "source_file"]

# keep META_COLS strictly numeric-only (or empty)
META_COLS = [
    # e.g. "time_idx", "elapsed_sec" ... (numeric only)
]
#Allowed values for categorical Features
#Important tip: keeping a list of allowed values for categorical features is great for sanity,
# but it doesn’t by itself wire the column into the pipeline.In fact,thses 3 lines are
#only for documentation because our context only flows parent → children;
# values seen in a child (e.g., inside hmd_data) never flow back up to the parent where you emit the row.

PHASE_VALUES = [
"intro", "talk_with_avatar", "calming_walk",
"normal_puzzle", "hard_puzzle", "impossible_puzzle"]
AREA_VALUES = [

"IntroArea", "PuzzleArea", "CalmingWalkArea"
]

GAZE_ACTOR = ["character_entity" , "interactive_entity" , "environment_entity"]

#End of Allowed values

# used by preprocess_and_tag_groups for contiguous-run grouping
GROUP_KEYS = ["phase", "area", "user_emotion"]

# Toggle grouped embedding on/off
USE_GROUPED_EMBEDDING = True  # or False for legacy

# Behavioral sub-groups (from your BEHAVIORAL_COLS)
BEHAVIORAL_GROUPS = {
    "gaze": ["gaze_x","gaze_y","gaze_z"],
    "head_pos": ["head_pos_x","head_pos_y","head_pos_z"],
    "r_pos": ["r_pos_x","r_pos_y","r_pos_z"],
    "l_pos": ["l_pos_x","l_pos_y","l_pos_z"],
    "speeds": ["movement_speed","r_movement_speed","l_movement_speed"],
}

# Output slice sizes per group; must sum to ATTN_CONFIG["d_model"]
GROUP_OUT_DIMS = {
    "gaze": 16,
    "head_pos": 16,
    "r_pos": 16,
    "l_pos": 16,
    "speeds": 16,
    "categoricals": 32,
    "text": 16,
    "meta": 0,
}

# ---------- Text / DistilBERT Hyper-parameters ----------
TEXT_USE_DEFAULT: bool = True          # master enable (pipeline can still auto-disable on 0% coverage)
SPEECH_COL: str = "text"

TEXT_MODEL_NAME: str = "distilbert-base-uncased"
TEXT_MAX_LEN: int = 64                 #Max tokens per sentence fed into DistilBERT (longer → more context, slower).
TEXT_POOLING: str = "cls"              # How to turn token outputs into one vector: *"cls": use the first token (standard BERT trick).  or  * "mean": average all tokens (masked).
TEXT_FREEZE: bool = True               #If True, DistilBERT weights don’t update (faster, safer with little data). Set False to fine-tune.
TEXT_BATCH_SIZE: int = 128              # equal to size of 'd_model'
TEXT_EMPTY_RATIO_CUTOFF: float = 1.0   # skip model when batch is fully empty (1.0 = 100%)
TEXT_SHOW_PROGRESS: bool = False        # tqdm + done message
TEXT_PROGRESS_DESC: str = "DistilBERT (text)"

# Hand-quality filtering
ONLY_VALID_HANDS = False   # if True, keep only rows where has_valid_hands == 1
HAND_COLS = [
    "r_pos_x", "r_pos_y", "r_pos_z",
    "l_pos_x", "l_pos_y", "l_pos_z",
]
def assert_required_columns(df) -> None:
    # text is the transcript/speech content field in your CSV
    needed = set(BEHAVIORAL_COLS + CATEGORICAL_COLS + ["timestamp","session_id","text"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def cast_categoricals(df) -> None:
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("string").fillna("NA")
