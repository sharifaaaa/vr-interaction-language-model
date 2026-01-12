# viz_embeddings.py
from __future__ import annotations
from typing import Optional, Sequence, Dict, Tuple, Iterable
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.manifold import TSNE

# UMAP is optional
try:
    import umap  # pip install umap-learn
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

def load_emotion_classifier(path: str) -> torch.nn.Module:
    """
    Loads the fine-tuned emotion classifier (Transformer + classification head).
    This function expects that the file is a full model (not just state_dict).

    Returns:
        model (torch.nn.Module)
    """
    print(f"🧩 Loading fine-tuned emotion classifier from: {path}")
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        print("✅ Classifier loaded as nn.Module directly.")
        return obj

    # Handle state_dict-only files
    if isinstance(obj, dict):
        from vr_transformer.transformer_classifier import TransformerClassifier
        try:
            model = TransformerClassifier()
            model.load_state_dict(obj.get("state_dict", obj), strict=False)
            print("✅ Classifier reconstructed from state_dict.")
            return model
        except Exception as e:
            print(f"⚠️ Could not rebuild TransformerClassifier from state_dict: {e}")
            return obj  # fallback

    raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")


# ===============================================================
# ---------------------- ENCODER LOADER -------------------------
# ===============================================================

def load_pretrained_encoder_from_config(CP, CA) -> Tuple[object, str, bool]:
    """
    Load a pretrained encoder according to config_pretrain + config_attention.

    - If FINETUNE_BACKBONE == "VRtransformer": use PRETRAIN_CKPT_PATH with '_vr' suffix (if present)
      and instantiate vr_transformer.transformer_blocks.TransformerEncoder() (no args).
    - If FINETUNE_BACKBONE == "temporal": instantiate train_pretrain.TemporalEncoder() (no args).

    Returns:
        encoder_or_obj: nn.Module if we could rebuild the model; otherwise raw checkpoint object
        ckpt_path:       path used
        is_module:       True if encoder_or_obj is an nn.Module
    """
    # Resolve checkpoint path based on backbone
    if CP.FINETUNE_BACKBONE == "VRtransformer":
        ckpt_path = CP.PRETRAIN_CKPT_PATH.replace(".pt", "_vr.pt")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ VRtransformer checkpoint not found at {ckpt_path}; falling back to base .pt")
            ckpt_path = CP.PRETRAIN_CKPT_PATH
    elif CP.FINETUNE_BACKBONE == "temporal":
        ckpt_path = CP.PRETRAIN_CKPT_PATH
    else:
        raise ValueError(f"Unknown FINETUNE_BACKBONE: {CP.FINETUNE_BACKBONE}")

    print(f"🧠 Loading {CP.FINETUNE_BACKBONE} encoder from: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")

    # If a whole module is stored, use it directly
    if isinstance(obj, nn.Module):
        print("✅ Loaded nn.Module directly from checkpoint.")
        return obj, ckpt_path, True

    # If we have a (nested) state_dict, rebuild a model using your no-arg constructors
    if isinstance(obj, dict):
        state_dict = obj.get("state_dict", obj)
        try:
            if CP.FINETUNE_BACKBONE == "VRtransformer":
                from vr_transformer.transformer_blocks import TransformerEncoder
                model = TransformerEncoder()   # uses global ATTN_CONFIG
            else:
                from train_pretrain import TemporalEncoder
                model = TemporalEncoder()      # uses internal defaults
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
            return model, ckpt_path, True
        except Exception as e:
            print(f"⚠️ Could not reconstruct model from state_dict: {e}")
            return obj, ckpt_path, False

    # Unknown type → return raw object; caller will fall back to token inputs
    print(f"⚠️ Unsupported checkpoint type: {type(obj)}. "
          f"Returning raw object; visualization will fall back to token inputs.")
    return obj, ckpt_path, False


# ===============================================================
# ------------------------ PLOTTING -----------------------------
# ===============================================================

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _legend_from_mapping(label_to_color: Dict[str, str]) -> Sequence[Patch]:
    return [Patch(facecolor=col, label=lab) for lab, col in label_to_color.items()]

def _colors_for_labels(
    labels: Sequence[str],
    label_to_color: Optional[Dict[str, str]]
) -> Sequence[str]:
    if label_to_color is None:
        cats = pd.Series(labels, dtype="category")
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in cats.cat.codes.values]
    return [label_to_color.get(str(l), "#999999") for l in labels]

def _save_scatter_2d(
    XY: np.ndarray,
    labels: Optional[Sequence[str]],
    out_png: str,
    title: str,
    label_to_color: Optional[Dict[str, str]],
) -> None:
    plt.figure(figsize=(7, 7))
    if labels is None:
        plt.scatter(XY[:, 0], XY[:, 1], s=10, alpha=0.9)
    else:
        cols = _colors_for_labels(labels, label_to_color)
        plt.scatter(XY[:, 0], XY[:, 1], c=cols, s=10, alpha=0.9)
        if label_to_color:
            plt.legend(handles=_legend_from_mapping(label_to_color),
                       loc="lower right", frameon=True, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def _save_scatter_3d(
    XYZ: np.ndarray,
    labels: Optional[Sequence[str]],
    out_png: str,
    title: str,
    label_to_color: Optional[Dict[str, str]],
) -> None:
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    if labels is None:
        ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], s=8, alpha=0.9)
    else:
        cols = _colors_for_labels(labels, label_to_color)
        ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], c=cols, s=8, alpha=0.9)
        if label_to_color:
            ax.legend(handles=_legend_from_mapping(label_to_color),
                      loc="lower right", frameon=True, fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ===============================================================
# --------------------- EMBEDDING COMPUTE -----------------------
# ===============================================================

@torch.no_grad()
def compute_chunk_embeddings(
    df: pd.DataFrame,
    embedder,
    get_chunk_batch_fn,          # e.g., get_chunk_batch(df, embedder, pad_to=...)
    pad_to: int,
    pooling: str = "mean",       # "mean" | "sum" | "cls"
    device: Optional[str] = None,
    label_col: Optional[str] = "user_emotion",
    encoder: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, Optional[Sequence[str]], Sequence[str]]:
    """
    Builds masked pooled embeddings per chunk.

    Returns:
        Z : (N, D) embeddings (token inputs if encoder=None, else encoder features)
        y : optional labels aligned to Z
        ids: chunk_id order used for Z
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, M, ids = get_chunk_batch_fn(df, embedder, pad_to=pad_to)  # X:(N,T,D), M:(N,T)
    X, M = X.to(device), M.to(device)

    # Use encoder if available; handle CLS (mask length +1) when POOLING="cls"
    if encoder is not None and isinstance(encoder, nn.Module):
        encoder = encoder.to(device).eval()

        # Detect if encoder inserts a CLS at the first layer
        try:
            uses_cls = (
                hasattr(encoder, "layers")
                and len(getattr(encoder, "layers", [])) > 0
                and hasattr(encoder.layers[0], "mha")
                and getattr(encoder.layers[0].mha, "use_cls", False)
            )
        except Exception:
            uses_cls = False

        M_for_encoder = M
        if uses_cls and M_for_encoder.dim() == 2:
            # pad mask on the left for CLS token: (N, T) -> (N, T+1)
            M_for_encoder = F.pad(M_for_encoder, (1, 0), value=1)

        # Forward (some encoders accept (X, mask), others only X)
        try:
            H = encoder(X, M_for_encoder)
        except TypeError:
            H = encoder(X)

        if isinstance(H, (tuple, list)):   # e.g., (scores, weights, Y)
            H = H[-1]

        # Pool with the mask that matches H's length
        M_pool = M_for_encoder if H.size(1) == M_for_encoder.size(1) else M
    else:
        if encoder is not None and not isinstance(encoder, nn.Module):
            print("⚠️ `encoder` is not an nn.Module (likely a raw object/state_dict). "
                  "Falling back to token inputs; load weights into a model to use learned features.")
        H = X
        M_pool = M

    # Masked pooling
    mask = M_pool.unsqueeze(-1).float()          # (N,T[, +1],1)
    masked = H * mask
    lengths = mask.sum(dim=1).clamp(min=1e-6)

    p = pooling.lower()
    if p in ("mean", "avg", "average"):
        pooled = masked.sum(dim=1) / lengths
    elif p == "sum":
        pooled = masked.sum(dim=1)
    elif p in ("cls", "first"):
        pooled = H[:, 0, :]
    else:
        pooled = masked.sum(dim=1) / lengths

    Z = pooled.detach().cpu().numpy()

    y = None
    if label_col and label_col in df.columns:
        id2lab = dict(zip(df["chunk_id"].astype(str), df[label_col].astype(str)))
        y = [id2lab.get(str(i), None) for i in ids]

    return Z, y, ids


# ===============================================================
# ----------------------- DIM. REDUCTION ------------------------
# ===============================================================

def _fit_tsne(
    Z: np.ndarray, n_components: int,
    perplexity: int, learning_rate: int, random_state: int
) -> np.ndarray:
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
    )
    return tsne.fit_transform(Z)

def _fit_umap(
    Z: np.ndarray, n_components: int,
    n_neighbors: int, min_dist: float, random_state: int
) -> Optional[np.ndarray]:
    if not _HAS_UMAP:
        print("ℹ️ UMAP not installed; skipping. `pip install umap-learn` to enable.")
        return None
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(Z)

def _export_csv(
    coords: np.ndarray, labels: Optional[Sequence[str]], ids: Sequence[str], path: str
) -> None:
    cols = ["x", "y"] if coords.shape[1] == 2 else ["x", "y", "z"]
    df = pd.DataFrame(coords, columns=cols)
    df["chunk_id"] = list(ids)
    if labels is not None:
        df["label"] = list(labels)
    df.to_csv(path, index=False)


def run_tsne_umap(
    Z: np.ndarray,
    labels: Optional[Sequence[str]],
    ids: Sequence[str],
    out_dir: str,
    prefix: str = "all",
    dims: Iterable[int] = (2, 3),          # which dimensions to produce
    tsne_perplexity: int = 30,
    tsne_learning_rate: int = 200,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
    label_to_color: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Produces t-SNE and (if available) UMAP in the specified dimensions.
    Saves CSVs and PNGs; returns a dict of artifact paths.
    """
    _ensure_dir(out_dir)
    arts: Dict[str, str] = {}

    for k in dims:
        # --- t-SNE ---
        XY = _fit_tsne(
            Z, n_components=k,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            random_state=random_state
        )
        csv_p = os.path.join(out_dir, f"{prefix}_tsne_{k}d.csv")
        _export_csv(XY, labels, ids, csv_p)
        png_p = os.path.join(out_dir, f"{prefix}_tsne_{k}d.png")
        if k == 2:
            _save_scatter_2d(XY, labels, png_p, f"t-SNE ({k}D) of Encoder Embeddings", label_to_color)
        else:
            _save_scatter_3d(XY, labels, png_p, f"t-SNE ({k}D) of Encoder Embeddings", label_to_color)
        arts[f"tsne_{k}d_csv"] = csv_p
        arts[f"tsne_{k}d_png"] = png_p

        # --- UMAP ---
        U = _fit_umap(
            Z, n_components=k,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state
        )
        if U is not None:
            csv_u = os.path.join(out_dir, f"{prefix}_umap_{k}d.csv")
            _export_csv(U, labels, ids, csv_u)
            png_u = os.path.join(out_dir, f"{prefix}_umap_{k}d.png")
            if k == 2:
                _save_scatter_2d(U, labels, png_u, f"UMAP ({k}D) of Encoder Embeddings", label_to_color)
            else:
                _save_scatter_3d(U, labels, png_u, f"UMAP ({k}D) of Encoder Embeddings", label_to_color)
            arts[f"umap_{k}d_csv"] = csv_u
            arts[f"umap_{k}d_png"] = png_u

    return arts


# ===============================================================
# -------------------------- DRIVER -----------------------------
# ===============================================================

def run_embedding_viz_from_df(
    chunked_df: pd.DataFrame,
    embedder,
    get_chunk_batch_fn,
    pad_to: int,
    out_dir: str,
    pooling: str = "mean",
    label_col: Optional[str] = "user_emotion",
    prefix: str = "chunks",
    dims: Iterable[int] = (2, 3),            # generate 2D and 3D by default
    label_to_color: Optional[Dict[str, str]] = None,
    split_filter: Optional[str] = None,      # e.g., "val" to use validation only
    max_points: Optional[int] = None,        # subsample for speed
    encoder: Optional[nn.Module] = None,     # learned encoder features (if provided)
) -> Dict[str, str]:
    """
    End-to-end:
      (optional split/filter) -> compute pooled embeddings (optionally via encoder)
      -> run t-SNE/UMAP in 2D/3D -> export CSVs/PNGs.
    """
    _ensure_dir(out_dir)

    df = chunked_df
    if split_filter is not None and "split" in df.columns:
        df = df[df["split"] == split_filter].reset_index(drop=True)

    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)

    Z, y, ids = compute_chunk_embeddings(
        df=df,
        embedder=embedder,
        get_chunk_batch_fn=get_chunk_batch_fn,
        pad_to=pad_to,
        pooling=pooling,
        label_col=label_col,
        encoder=encoder,
    )
    print(f"🔎 Embeddings: {Z.shape[0]} chunks × {Z.shape[1]} dims (split={split_filter or 'ALL'})")

    return run_tsne_umap(
        Z=Z, labels=y, ids=ids,
        out_dir=out_dir, prefix=prefix, dims=dims,
        label_to_color=label_to_color,
    )
# --- put near other imports in viz_embeddings.py ---
import os
from torch import nn

# (Assumes these helpers already exist in viz_embeddings.py)
# - load_pretrained_encoder_from_config
# - load_emotion_classifier (if you added it earlier)
# - run_embedding_viz_from_df

def visualize_encoder_embeddings(
    chunked_df,
    embedder,
    get_chunk_batch_fn,
    CA, CP,
    out_dir: str | None = None,
    label_to_color: dict[str, str] | None = None,
    split_filter: str | None = "val",
    max_points: int | None = 5000,
    dims: tuple[int, ...] = (2, 3),
) -> dict[str, str]:
    """
    Visualize embeddings using the PRETRAINED ENCODER (from CP/CA).
    Returns a dict of artifact paths (PNGs/CSVs).
    """
    out_dir = out_dir or "./artifacts/embed_viz"
    os.makedirs(out_dir, exist_ok=True)

    # default color mapping
    if label_to_color is None:
        label_to_color = {
            "joy": "#1f77b4",
            "neutral": "#7f7f7f",
            "frustration": "#d62728",
            "stress": "#2ca02c",
        }

    # Load encoder (module or raw object)
    encoder_obj, ckpt_path, is_module = load_pretrained_encoder_from_config(CP, CA)
    print(f"✅ Encoder object loaded from {ckpt_path} (nn.Module={is_module})")
    encoder = encoder_obj if isinstance(encoder_obj, nn.Module) else None

    paths = run_embedding_viz_from_df(
        chunked_df=chunked_df,
        embedder=embedder,
        get_chunk_batch_fn=get_chunk_batch_fn,
        pad_to=CA.ATTN_CONFIG["max_len"],
        out_dir=out_dir,
        pooling=CA.POOLING,
        label_col="user_emotion",
        prefix="val",
        dims=dims,
        label_to_color=label_to_color,
        split_filter=split_filter,
        max_points=max_points,
        encoder=encoder,  # None if reconstruction failed → uses token inputs
    )

    print("🖼 Embedding viz artifacts:")
    for k, v in paths.items():
        print("  ", k, "->", v)
    return paths


def visualize_classifier_embeddings(
    chunked_df,
    embedder,
    get_chunk_batch_fn,
    CA,
    classifier_path: str = "emotion_classifier.pt",
    out_dir: str | None = None,
    label_to_color: dict[str, str] | None = None,
    split_filter: str | None = "val",
    max_points: int | None = 5000,
    dims: tuple[int, ...] = (2, 3),
) -> dict[str, str]:
    """
    Visualize embeddings using the FINE-TUNED CLASSIFIER as the encoder.
    Returns a dict of artifact paths (PNGs/CSVs).
    """
    out_dir = out_dir or "./artifacts/embed_viz_classifier"
    os.makedirs(out_dir, exist_ok=True)

    if label_to_color is None:
        label_to_color = {
            "joy": "#1f77b4",
            "neutral": "#7f7f7f",
            "frustration": "#d62728",
            "stress": "#2ca02c",
        }

    if not os.path.exists(classifier_path):
        print(f"⚠️ No fine-tuned classifier found at: {classifier_path}")
        return {}

    # You added this in a prior step; it returns a Module or raw object
    clf_model = load_emotion_classifier(classifier_path)
    encoder = clf_model if isinstance(clf_model, nn.Module) else None

    paths = run_embedding_viz_from_df(
        chunked_df=chunked_df,
        embedder=embedder,
        get_chunk_batch_fn=get_chunk_batch_fn,
        pad_to=CA.ATTN_CONFIG["max_len"],
        out_dir=out_dir,
        pooling=CA.POOLING,
        label_col="user_emotion",
        prefix="val_clf",
        dims=dims,
        label_to_color=label_to_color,
        split_filter=split_filter,
        max_points=max_points,
        encoder=encoder,  # uses classifier backbone features
    )

    print("\n📊 Classifier embedding viz artifacts:")
    for k, v in paths.items():
        print("  ", k, "->", v)
    return paths
def run_all_embedding_visualizations(
    chunked_df,
    embedder,
    get_chunk_batch_fn,
    CA,
    CP,
    classifier_path: str = "emotion_classifier.pt",
    out_root: str = "./artifacts/all_viz",
    max_points: int | None = 5000,
    dims: tuple[int, ...] = (2, 3),
    label_to_color: dict[str, str] | None = None,
    split_filter: str | None = "val",
) -> dict[str, dict[str, str]]:
    """
    Run all three embedding visualizations:
      1. Raw token-level embeddings (no encoder)
      2. Pretrained encoder embeddings
      3. Fine-tuned classifier embeddings

    Returns a nested dict:
      {
        "baseline": {...artifact paths...},
        "encoder": {...artifact paths...},
        "classifier": {...artifact paths...}
      }
    """
    os.makedirs(out_root, exist_ok=True)

    # --- 1. Baseline (token-level) ---
    print("\n=== 🔹 BASELINE: Raw Token Embeddings ===")
    baseline_out = os.path.join(out_root, "baseline")
    paths_baseline = run_embedding_viz_from_df(
        chunked_df=chunked_df,
        embedder=embedder,
        get_chunk_batch_fn=get_chunk_batch_fn,
        pad_to=CA.ATTN_CONFIG["max_len"],
        out_dir=baseline_out,
        pooling=CA.POOLING,
        label_col="user_emotion",
        prefix="val_base",
        dims=dims,
        label_to_color=label_to_color,
        split_filter=split_filter,
        max_points=max_points,
        encoder=None,
    )

    # --- 2. Encoder embeddings ---
    print("\n=== 🔹 PRETRAINED ENCODER EMBEDDINGS ===")
    encoder_out = os.path.join(out_root, "encoder")
    paths_encoder = visualize_encoder_embeddings(
        chunked_df=chunked_df,
        embedder=embedder,
        get_chunk_batch_fn=get_chunk_batch_fn,
        CA=CA,
        CP=CP,
        out_dir=encoder_out,
        label_to_color=label_to_color,
        split_filter=split_filter,
        max_points=max_points,
        dims=dims,
    )

    # --- 3. Classifier embeddings ---
    print("\n=== 🔹 FINE-TUNED CLASSIFIER EMBEDDINGS ===")
    classifier_out = os.path.join(out_root, "classifier")
    paths_classifier = visualize_classifier_embeddings(
        chunked_df=chunked_df,
        embedder=embedder,
        get_chunk_batch_fn=get_chunk_batch_fn,
        CA=CA,
        classifier_path=classifier_path,
        out_dir=classifier_out,
        label_to_color=label_to_color,
        split_filter=split_filter,
        max_points=max_points,
        dims=dims,
    )

    print("\n✅ All embedding visualizations completed.")
    return {
        "baseline": paths_baseline,
        "encoder": paths_encoder,
        "classifier": paths_classifier,
    }

