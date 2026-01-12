# main_emotionRecognition.py
from __future__ import annotations
from typing import Tuple, Optional
import os
import pandas as pd
import torch

# --- local modules ---
from preprocess_vr_data import preprocess_and_tag_groups, chunk_grouped_data
from token_embedder import (
    TokenEmbedder, TokenEmbedderConfig,
    build_categorical_vocabs, save_vocabs, load_vocabs
)
import config_attention as CA
import config_pretrain as CP
from vr_transformer.attention_runner import run_attention, get_chunk_batch
from classifier_runner import run_classifier
from vr_transformer.transformer_classifier import TransformerClassifier
from train_classifier import train_transformer_classifier
from utils_labels import labeled_chunk_ids, build_label_mapping, labels_tensor
from merge_sessions import merge_jsons_to_featurevector
from config_features import (
    BEHAVIORAL_COLS, CATEGORICAL_COLS, META_COLS, SPEECH_COL,
    CAT_EMBED_DIM, assert_required_columns, GROUP_OUT_DIMS,ONLY_VALID_HANDS
)
from sessions_fetcher import fetch_sessions_with_fallback
from pretrain_dataset import make_unlabeled_windows_dataset, make_labeled_windows_dataset, labeled_collate_fn
from train_pretrain import run_temporal_pretraining, run_vr_pretraining
from train_finetune import run_supervised_finetune
from timeline_report import generate_timeline, plot_timeline_stripes_time
from emotion_visuals import generate_all_emotion_visuals
from text_projection_dump import dump_text_projections
from Audio_Branch.fusion_with_audio import run_fusion_with_audio

# master-table (SQLite) + policy
from db_sessions import init_db, register_scanned_sessions
from policy_manager import (
    DataPolicy, apply_allowed_modes, select_paths_for_mode, print_policy_and_counts
)

from evaluate_only import evaluate_dataset
from av_plotter import plot_av_trajectory_per_session, plot_av_session_mean
# TRAIN-only preprocessing helpers
from prep_fit import fit_numeric_scaler, apply_numeric_scaler, save_scaler, load_scaler
from viz_embeddings import run_all_embedding_visualizations
from utils_splits import (
    load_stratified_mapping, apply_split_override,
    assert_split_consistency, report_split_status, auto_session_col
)

# ---------- paths for artifacts ----------
ART_DIR      = "./artifacts"
SCALER_PATH  = os.path.join(ART_DIR, "scaler.pkl")
VOCABS_PATH  = os.path.join(ART_DIR, "cat_vocabs.json")
TEST_OUT_DIR = os.path.join(ART_DIR, "test_eval")   # base dir for evaluate Mode's artifacts


def _save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"✅ DataFrame saved to {path}")


def _safe_text_coverage(df: pd.DataFrame) -> float:
    col = "text" if "text" in df.columns else None
    if not col:
        return 0.0
    s = df[col].fillna("").astype(str).str.strip()
    return float((s != "").mean())


def _ensure_art_dir():
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(TEST_OUT_DIR, exist_ok=True)  # ensure evaluate folder exists


def run_pipeline_multi() -> Tuple[Optional[pd.DataFrame], Optional[TokenEmbedder]]:
    print("🚀 Starting pipeline...")
    _ensure_art_dir()

    # ---- 1) Fetch & register
    used_dir, remote_ok, new_files = fetch_sessions_with_fallback()
    init_db()
    source = "remote" if remote_ok else "local"
    try:
        registered = register_scanned_sessions(used_dir, source=source)
    except Exception as e:
        print(f"⚠️ Failed to register scanned sessions ({e}). Proceeding with folder scan only.")
        registered = 0
    print(f"🗄️  Registered/updated {registered} sessions in SQLite ({source}).")

    # ---- 2) Policy & selection
    policy = DataPolicy(
        use_labeled_in_pretrain=True,
        unlabeled_both_except_finetune_selected=True,
        pretrain_splits_allowed=("train", "val"),
        finetune_splits_allowed=("train", "val"),
        stratified_ratio=(0.8, 0.1, 0.1),
        stratify_by="dominant_label",
        stratify_scope="labeled",
        stratify_seed=42,
        stratify_when="always"  #"if_db_missing"
    )
    #from db_sessions import mark_finetune_selection
    #mark_finetune_selection(["session_abc", "session_def"], lock=True)
    apply_allowed_modes(policy)
    print_policy_and_counts(policy)

    Mode = CA.MODE


    # ---- 3) Select and merge
    try:
        if Mode in ("pretrain", "finetune", "interpret", "evaluate","fusion_with_audio","embed-viz"):
            mode_for_db = (
                "pretrain" if Mode == "pretrain"
                else ("finetune" if Mode in ("finetune", "interpret","embed-viz") else "evaluate")
            )

            paths, used_strat = select_paths_for_mode(mode_for_db, policy, base_dir=used_dir)

            if not paths:
                print(f"⚠️ No sessions selected by policy for MODE={Mode}. Falling back to folder scan.")
                merge_jsons_to_featurevector(data_dir=used_dir, out_csv="featureVector_original.csv")
            else:
                merge_jsons_to_featurevector(file_list=paths, out_csv="featureVector_original.csv")
                origin = "stratified split" if used_strat else "DB master table"
                print(f"📂 Selected {len(paths)} files for MODE={Mode} via {origin}.")
        else:
            merge_jsons_to_featurevector(data_dir=used_dir, out_csv="featureVector_original.csv")
            print(f"📂 Selected files by scanning folder (no DB) for MODE={Mode}.")


    except Exception as e:
        print(f"⚠️ Merge failed ({e}). Trying raw folder scan fallback…")
        merge_jsons_to_featurevector(data_dir=used_dir, out_csv="featureVector_original.csv")

    # Sanity check minimum columns
    df_check = pd.read_csv("featureVector_original.csv", low_memory=False)
    #Hand Quality- How many sessions are good vs bad?
    print(df_check.groupby("has_valid_hands")["db_session_id"].nunique())
    # Rough percentage of good vs bad rows
    print(df_check["has_valid_hands"].value_counts(normalize=True))
    #--End of Hand quality
    req = {"db_session_id", "split"}
    if not req.issubset(df_check.columns):
        print(f"⚠️ featureVector_original.csv missing {req - set(df_check.columns)}; downstream will degrade.")
    else:
        print(df_check[["db_session_id", "split"]].drop_duplicates().head())

    # ---- 4) Preprocess (stateless)
    grouped_df = preprocess_and_tag_groups("featureVector_original.csv", output_path="tagged_vr_data.csv")

    # ---- 5) Chunk
    chunk_size = CA.ATTN_CONFIG["max_len"]
    stride     = CA.ATTN_CONFIG["stride"]
    chunked_df = chunk_grouped_data(grouped_df, chunk_size=chunk_size, stride=stride)
    #Debug
    #Reduce scope to test:
    #chunked_df = chunked_df.head(50_000)  # TEMP
    # Add flush=True to key prints so you see progress immediately.
    _save_df(chunked_df, "chunked_vr_data.csv")
    print("✅ Saved chunked_vr_data.csv", flush=True)
    #End of Debug

    #override value of 'split' for each session on static DB split in CSVs when stratified split is active
    if used_strat:

        mp = load_stratified_mapping("./artifacts/stratified_splits/latest.csv")

        for path in ["featureVector_original.csv", "tagged_vr_data.csv", "chunked_vr_data.csv"]:
            df = pd.read_csv(path, low_memory=False)
            sc = auto_session_col(df)  # will pick db_session_id if present
            df = apply_split_override(
                df, mp, session_col=sc,
                strict=False,
                # when your ratio is (0.8, 0.2, 0.0), force any non-mapped rows to 'val'
                default_for_missing=("val" if True else None)
            )
            assert_split_consistency(df, mp, session_col=sc)
            df.to_csv(path, index=False)
            print("✅ Rewrote", path)

        # refresh in-memory DataFrames
        fv = pd.read_csv("featureVector_original.csv", low_memory=False)
        grouped_df = pd.read_csv("tagged_vr_data.csv", low_memory=False)
        chunked_df = pd.read_csv("chunked_vr_data.csv", low_memory=False)

        # Coverage and split summaries
        report_split_status("featureVector_original.csv", fv, mp)
        report_split_status("tagged_vr_data.csv", grouped_df, mp)
        report_split_status("chunked_vr_data.csv", chunked_df, mp)

    #End of overrid

    # ---- Summary
    print("\n📈 Summary")
    print(f"• Source: {'remote API' if remote_ok else 'local folder'} ({used_dir})")
    print(f"• New files downloaded this run: {new_files}")
    print(f"• Total original records: {len(grouped_df)}")
    print(f"• Total records after chunking: {len(chunked_df)}")
    print(f"• Total unique groups: {grouped_df['group_id'].nunique()}")
    print(f"• Total unique chunks: {chunked_df['chunk_id'].nunique()}")
    head_cols = [c for c in ("chunk_id", "group_id", "user_emotion") if c in chunked_df.columns]
    print(chunked_df[head_cols].head())


    # ---- 6) Split frames
    if "split" not in chunked_df.columns:
        print("⚠️ No 'split' column found; assigning all rows to 'train'.")
        chunked_df = chunked_df.copy()
        chunked_df["split"] = "train"

    # Optionally filter by hand-tracking quality BEFORE making splits

    use_only_valid = ONLY_VALID_HANDS  #from config_features
    df_for_split = chunked_df

    if use_only_valid:
        if "has_valid_hands" in df_for_split.columns:
            before = len(df_for_split)
            df_for_split = df_for_split[df_for_split["has_valid_hands"] == 1].reset_index(drop=True)
            removed = before - len(df_for_split)
            print(f"🧹 ONLY_VALID_HANDS=True → kept {len(df_for_split)} rows "
                  f"(removed {removed} rows with has_valid_hands=0).")
        else:
            print("⚠️ ONLY_VALID_HANDS=True but 'has_valid_hands' column is missing. "
                  "Skipping quality-based filtering.")
            df_for_split = df_for_split.reset_index(drop=True)
    else:
        df_for_split = df_for_split.reset_index(drop=True)

    # Now build splits from the (maybe filtered) DataFrame
    train_df = df_for_split[df_for_split["split"] == "train"].reset_index(drop=True)
    val_df = df_for_split[df_for_split["split"] == "val"].reset_index(drop=True)
    test_df = df_for_split[df_for_split["split"] == "test"].reset_index(drop=True)

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ---- 7) Fit/apply scaler + vocabs
    vocabs = None
    if Mode == "evaluate":
        try:
            scaler = load_scaler(SCALER_PATH)
            print(f"🔁 Loaded scaler from {SCALER_PATH}")
        except Exception:
            print("⚠️ Missing scaler.pkl — evaluating with scaler fit on TEST (demo-only).")
            scaler = fit_numeric_scaler(test_df, BEHAVIORAL_COLS)
        test_df = apply_numeric_scaler(test_df, BEHAVIORAL_COLS, scaler)

        try:
            vocabs = load_vocabs(VOCABS_PATH)
            print(f"🔁 Loaded categorical vocabs from {VOCABS_PATH}")
        except Exception:
            print("⚠️ Missing cat_vocabs.json — building vocabs from TEST (demo-only).")
            vocabs = build_categorical_vocabs(test_df, CATEGORICAL_COLS)
    else:
        if len(train_df) == 0:
            print("⚠️ No TRAIN rows. Fitting scaler on VAL or ALL as fallback.")
        fit_base = train_df if len(train_df) > 0 else (val_df if len(val_df) > 0 else chunked_df)

        scaler = fit_numeric_scaler(fit_base, BEHAVIORAL_COLS)
        train_df = apply_numeric_scaler(train_df, BEHAVIORAL_COLS, scaler)
        val_df   = apply_numeric_scaler(val_df,   BEHAVIORAL_COLS, scaler)
        test_df  = apply_numeric_scaler(test_df,  BEHAVIORAL_COLS, scaler)
        try:
            save_scaler(scaler, path=SCALER_PATH)
            print(f"💾 Saved scaler to {SCALER_PATH}")
        except Exception as e:
            print(f"⚠️ Could not save scaler: {e}")

        try:
            vocabs = build_categorical_vocabs(fit_base, CATEGORICAL_COLS)
            save_vocabs(vocabs, VOCABS_PATH)
            print(f"💾 Saved categorical vocabs to {VOCABS_PATH}")
        except Exception as e:
            print(f"⚠️ Could not build/save vocabs ({e}). Building from ALL as last resort.")
            vocabs = build_categorical_vocabs(chunked_df, CATEGORICAL_COLS)

    # ---- 8) Embedder
    text_coverage = _safe_text_coverage(chunked_df)
    USE_TEXT = bool(text_coverage > 0.00001)
    #USE_TEXT = False
    print(f"🗣️ Text coverage: {text_coverage:.1%} → use_text={USE_TEXT}")

    n_rows = len(chunked_df)
    n_txt = chunked_df["text"].astype(str).str.strip().replace("none", "").ne("").sum()
    print(f"Text coverage: {n_txt}/{n_rows} = {n_txt / n_rows:.3%}")


    embed_cfg = TokenEmbedderConfig(
        behavioral_cols=BEHAVIORAL_COLS,
        categorical_cols=CATEGORICAL_COLS,
        meta_cols=META_COLS,
        cat_embed_dim=CAT_EMBED_DIM,
        proj_dim=CA.ATTN_CONFIG["d_model"],
        use_text=USE_TEXT,
        speech_col=SPEECH_COL,
    )
    embedder = TokenEmbedder(embed_cfg, vocabs or {})
    print("✅ Embedder ready.")
    print(f"✅ Transformer uses pos_type={CA.ATTN_CONFIG.get('pos_type','(n/a)')}.")

    # Optional: text projection
    try:
        if USE_TEXT and GROUP_OUT_DIMS.get("text", 0) > 0:
            dump_text_projections(chunked_df, embedder, out_csv="chunked_with_textProj.csv")

        else:
            print("ℹ️ Text projection dump skipped (use_text=False or output dim=0).")
    except Exception as e:
        print(f"⚠️ Text projection dump skipped due to error: {e}")

    try:
        torch.save(embedder, "embedder.pt")
        print("💾 Saved embedder to embedder.pt")
    except Exception as e:
        print(f"⚠️ Could not save embedder: {e}")

    # ---- 9) Route by mode
    pooling = CA.POOLING
    print(f"\n🔧 Config -> MODE: {Mode}, POOLING: {pooling}")

    try:
        if Mode in ("single", "multi", "batch", "enc-single", "enc-batch"):
            run_attention(Mode, chunked_df, embedder)

        elif Mode in ("clf-single", "clf-batch"):
            run_classifier(
                Mode, chunked_df, embedder,
                print_pooled=globals().get("PRINT_POOLED", False),
                max_print=globals().get("MAX_PRINT", 10)
            )

        elif Mode == "train-clf":
            keep_ids = labeled_chunk_ids(chunked_df, "user_emotion")
            if not keep_ids:
                print("⚠️ No labeled chunks found. Skipping train-clf.")
            else:
                df_labeled = chunked_df[chunked_df["chunk_id"].isin(keep_ids)]
                X, M, ids = get_chunk_batch(df_labeled, embedder, pad_to=CA.ATTN_CONFIG["max_len"])
                class_to_idx = build_label_mapping(df_labeled, "user_emotion")
                y = labels_tensor(df_labeled, ids, class_to_idx)
                print(f"📊 Classes: {class_to_idx} | kept chunks: {len(ids)}")
                X, M, y = X.detach(), M.detach(), y.detach()

                model = TransformerClassifier(num_classes=len(class_to_idx))
                print(f"\n🧪 Training config -> batch={CP.BATCH_SIZE}, epochs={CP.EPOCHS}, "
                      f"lr={CP.LR}, wd={CP.WEIGHT_DECAY}, val={CP.VAL_RATIO}, seed={CP.SEED}")

                _ = train_transformer_classifier(
                    model, X, M, y,
                    class_to_idx=class_to_idx,
                    tracker_out_dir="./cm_plots"
                )

        elif Mode == "pretrain":
            if len(train_df) == 0:
                print("⚠️ pretrain: no TRAIN rows; using ALL rows as a fallback (demo-only).")
            ds_train = make_unlabeled_windows_dataset(
                train_df if len(train_df) else chunked_df,
                embedder=embedder,
                mask_scheme=CP.MASK_TOKEN_SCHEME,
                augs=CP.AUGS,
                pooling=CA.POOLING
            )
            ds_val = make_unlabeled_windows_dataset(
                val_df if len(val_df) else train_df,
                embedder=embedder,
                mask_scheme=CP.MASK_TOKEN_SCHEME,
                augs=CP.AUGS,
                pooling=CA.POOLING
            ) if len(val_df) > 0 else None

            n_val_info = (len(ds_val) if ds_val is not None else 0)
            print(f"🧪 Pretraining on {len(ds_train)} train windows and {n_val_info} val windows "
                  f"(objective: mtm={CP.OBJECTIVES.get('mtm',False)}, "
                  f"contrastive={CP.OBJECTIVES.get('contrastive',False)}, "
                  f"xmodal={CP.OBJECTIVES.get('xmodal',False)})")

            if CP.FINETUNE_BACKBONE == "temporal":
                run_temporal_pretraining(ds_train,used_stratified=used_strat) if ds_val is None else run_temporal_pretraining(ds_train, val_dataset=ds_val,used_stratified=used_strat)
            else:
                run_vr_pretraining(ds_train,used_stratified=used_strat) if ds_val is None else run_vr_pretraining(ds_train, val_dataset=ds_val,used_stratified=used_strat)

        elif Mode == "finetune":
            if len(train_df) == 0 or "user_emotion" not in train_df.columns:
                print("⚠️ Finetune skipped: no TRAIN rows or labels present.")
            else:
                ds_train, class_to_idx = make_labeled_windows_dataset(
                    train_df, embedder=embedder, label_col="user_emotion"
                )
                ds_val, _ = make_labeled_windows_dataset(
                    val_df, embedder=embedder, label_col="user_emotion"
                ) if len(val_df) > 0 else (None, None)

                print(f"🧪 Fine-tuning on {len(ds_train)} train windows | classes: {class_to_idx}")
                if ds_val is not None:
                    run_supervised_finetune(dataset=ds_train, class_to_idx=class_to_idx, ds_val=ds_val)
                else:
                    run_supervised_finetune(dataset=ds_train, class_to_idx=class_to_idx)

        elif Mode == "interpret":
            from interpretability_captum import run_captum_report
            for target in (("pred",), ("gt",), ("joy", "pred")):
                try:
                    run_captum_report(
                        chunked_df, embedder,
                        label_col="user_emotion",
                        clf_ckpt_path=CP.CLF_CKPT_PATH,
                        out_dir="./artifacts/Captum_interpretation",
                        n_steps=64,
                        target_modes=target
                    )
                except Exception as e:
                    print(f"⚠️ Captum report {target} skipped: {e}")

        elif Mode == "evaluate":

            if len(test_df) == 0:
                print("⚠️ No TEST rows (split=='test'). Nothing to evaluate.")
            elif "user_emotion" not in test_df.columns:
                print("⚠️ TEST rows have no ground-truth labels; skipping metrics.")
            else:
                ds_test, class_to_idx = make_labeled_windows_dataset(
                    test_df, embedder=embedder, label_col="user_emotion"
                )

                # 1) Metrics + per-chunk predictions
                try:

                    _ = evaluate_dataset(
                        dataset=ds_test,
                        class_to_idx=class_to_idx,
                        ckpt_path=CP.CLF_CKPT_PATH,
                        out_dir=TEST_OUT_DIR,
                        batch_size=CP.BATCH_SIZE
                    )
                except Exception as e:
                    print(f"⚠️ Evaluation failed: {e}")

                # 2) Timeline & plot (saved under ./artifacts/test_eval)
                try:
                    print("\n🧭 Building timeline (ground-truth vs predictions)...")
                    timeline = generate_timeline(
                        chunked_df="chunked_vr_data.csv",
                        label_col="user_emotion",
                        ckpt_path=CP.CLF_CKPT_PATH,
                        embedder=embedder,
                        batch_size=CP.BATCH_SIZE,
                    )
                    timeline_csv = os.path.join(TEST_OUT_DIR, "timeline_emotions.csv")
                    timeline.to_csv(timeline_csv, index=False)
                    print(f"💾 Wrote {timeline_csv}")

                    plot_timeline_stripes_time(
                        timeline,
                        unit="s",
                        save_path=os.path.join(TEST_OUT_DIR, "timeline_emotions.png"),
                        title="VR session — emotions over time",
                        show=False,
                    )
                    print("🖼 Saved timeline plot in test_eval")
                except Exception as e:
                    print(f"⚠️ Timeline generation skipped: {e}")

                # 3) Emotion visuals (saved under ./artifacts/test_eval/plot_emotion_spectrum)
                try:
                    emotions_order = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
                    out_vis = os.path.join(TEST_OUT_DIR, "plot_emotion_spectrum")
                    outputs = generate_all_emotion_visuals(
                        timeline_csv=timeline_csv,
                        chunked_df="chunked_vr_data.csv",
                        clf_ckpt_path=CP.CLF_CKPT_PATH,
                        embedder=embedder,
                        emotions=emotions_order,
                        out_dir=out_vis,
                    )
                    for sid, paths in outputs.items():
                        print(f"[{sid}] visuals:")
                        for k, p in paths.items():
                            print("  ", k, "->", p)
                except Exception as e:
                    print("⚠️ Visual generation skipped:", e)

                # 4) AV plots (saved under ./artifacts/test_eval/av_plots/...)
                try:

                    traj_dir = os.path.join(TEST_OUT_DIR, "av_plots", "trajectory")
                    mean_dir = os.path.join(TEST_OUT_DIR, "av_plots", "mean")
                    av_traj_paths = plot_av_trajectory_per_session(
                        csv_path=timeline_csv,
                        out_dir=traj_dir,
                        label_col="pred_emotion",
                    )
                    av_mean_paths = plot_av_session_mean(
                        csv_path=timeline_csv,
                        out_dir=mean_dir,
                        label_col="pred_emotion",
                    )
                    print(f"AV trajectory plots: {len(av_traj_paths)} files")
                    print(f"AV mean plots:       {len(av_mean_paths)} files")
                except Exception as e:
                    print("⚠️ AV plotting skipped:", e)


        elif Mode == "fusion_with_audio":

            timeline_csv = os.path.abspath("artifacts/test_eval/timeline_emotions.csv")
            out_dir = os.path.abspath("artifacts/fusion_with_audio")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, "timeline_emotions_with_audio.csv")

            # reuse your mapping (already computed earlier in your main) ---
            keep_ids = labeled_chunk_ids(chunked_df, "user_emotion")
            if not keep_ids:
                print("⚠️ No labeled chunks found. Skipping train-clf.")
            else:
                df_labeled = chunked_df[chunked_df["chunk_id"].isin(keep_ids)]
                class_to_idx = build_label_mapping(df_labeled, "user_emotion")

            audio_model_path = "audio_cnn.pt"

            _ = run_fusion_with_audio(
                timeline_csv=timeline_csv,
                out_csv=out_csv,
                class_to_idx=class_to_idx,
                audio_model_path=audio_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                T_fixed=192,
                override_threshold=0.60,
            )

        ###


        elif Mode == "embed-viz":
            out_viz_dir = os.path.join(ART_DIR, "embed_viz")

            # Fixed legend colors (edit to match your labels)
            emotion_colors = {
                "joy": "#1f77b4",
                "neutral": "#7f7f7f",
                "frustration": "#d62728",
                "stress": "#2ca02c",
            }

            paths = run_all_embedding_visualizations(
                chunked_df=chunked_df,
                embedder=embedder,
                get_chunk_batch_fn=get_chunk_batch,
                CA=CA,
                CP=CP,
                classifier_path="emotion_classifier.pt",
                out_root=out_viz_dir,
                max_points=5000,
                dims=(2, 3),
                label_to_color=emotion_colors,
                split_filter="train", #"val",
            )

            print("\n🎯 All visualization artifacts:")
            for viz_type, result in paths.items():
                print(f"\n--- {viz_type.upper()} ---")
                for k, v in result.items():
                    print(" ", k, "->", v)


        else:
            print(f"⚠️ Unknown MODE='{Mode}'. Nothing executed.")

    except Exception as e:
        print(f"❗ Unhandled error while running mode '{Mode}': {e}")

    return chunked_df, embedder


if __name__ == "__main__":
    run_pipeline_multi()
