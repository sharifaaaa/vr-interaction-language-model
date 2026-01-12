# classifier_runner.py
from __future__ import annotations
import torch
from typing import Dict, List
import config_attention as CA
from vr_transformer.attention_runner import get_chunk_batch
from vr_transformer.transformer_classifier import TransformerClassifier
from utils_labels import build_label_mapping, labeled_chunk_ids  # <-- add this import


def _labels_tensor(df, ids: List[str], class_to_idx: Dict[str, int],
                   label_col: str = "user_emotion") -> torch.Tensor:
    y = []
    for cid in ids:
        lab = df.loc[df["chunk_id"] == cid, label_col].iloc[0]
        y.append(class_to_idx[str(lab)])  # lab will be a valid string after filtering
    return torch.tensor(y, dtype=torch.long)


def run_classifier(mode: str, chunked_df, embedder,
                   *, print_pooled: bool = False, max_print: int = 5) -> None:
    """
    Run the TransformerClassifier in:
      - 'clf-single' : first labeled chunk only
      - 'clf-batch'  : all labeled chunks
    """
    label_col = "user_emotion"
    pad_to = CA.ATTN_CONFIG["max_len"]

    # --- keep ONLY labeled chunks ---
    keep_ids = labeled_chunk_ids(chunked_df, label_col)
    if not keep_ids:
        raise RuntimeError("No labeled chunks found. Cannot run classifier with ground-truth.")
    df_lab = chunked_df[chunked_df["chunk_id"].isin(keep_ids)].copy()

    # class map from labeled-only frame
    class_to_idx = build_label_mapping(df_lab, label_col)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = TransformerClassifier(num_classes=num_classes)
    model.eval()

    if mode == "clf-single":
        cid, g = next(iter(df_lab.groupby("chunk_id", sort=False)))
        x, m = embedder(g, pad_to=pad_to)
        x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)   # (1,L,E)
        m = torch.as_tensor(m, dtype=torch.bool).unsqueeze(0)      # (1,L)

        with torch.no_grad():
            logits, pooled = model(x, m)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()

        print(f"\n--- CLASSIFIER for chunk {cid} ---")
        print("Probabilities:", probs.squeeze(0).tolist())
        print("Predicted:", idx_to_class[pred])
        if print_pooled:
            print("Pooled vector:", pooled.squeeze(0).detach().cpu().numpy())

    elif mode == "clf-batch":
        X, M, ids = get_chunk_batch(df_lab, embedder, pad_to=pad_to)   # <-- use df_lab
        y_true = _labels_tensor(df_lab, ids, class_to_idx, label_col=label_col)

        with torch.no_grad():
            logits, pooled = model(X, M)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        acc = (preds == y_true).float().mean().item()
        print(f"\n--- CLASSIFIER BATCH for {len(ids)} labeled chunks ---  acc={acc:.3f}")

        for i, cid in enumerate(ids[:max_print]):
            print(f"- {cid}: true={idx_to_class[y_true[i].item()]}"
                  f" | pred={idx_to_class[preds[i].item()]}"
                  f" | probs={probs[i].tolist()}")
            if print_pooled:
                print("  pooled:", pooled[i].detach().cpu().numpy())
    else:
        raise ValueError("Invalid mode. Use 'clf-single' or 'clf-batch'.")
