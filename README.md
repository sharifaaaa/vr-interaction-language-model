# VR Interaction as a Language-Like Sequence

This repository contains research code for modeling **embodied interaction in Virtual Reality (VR)** as **language-like sequential data**, using Transformer-based architectures.

The core idea is to treat continuous VR interaction logs (gaze, motion, actions) as structured temporal sequences, analogous to natural language, enabling downstream tasks such as **emotion recognition** and **cognitive state inference**.

---

## Research Motivation

Human behavior in immersive environments unfolds over time through embodied interaction.  
While most VR analytics rely on handcrafted features or static aggregation, this project explores whether **sequential modeling techniques from NLP** can be applied directly to **VR behavioral streams**.

Key research questions:
- Can VR interaction traces be treated as a tokenized sequence?
- Do Transformer models capture temporal dependencies in embodied behavior?
- How well can emotion states be inferred from non-verbal interaction alone?

This work supports ongoing research in **Human–Computer Interaction (HCI)**, **XR**, and **affective computing**.

---

## Method Overview

1. **Raw VR logs** (gaze, head, hand motion, events)  
2. **Preprocessing & temporal chunking**
3. **Feature projection into sequence representations**
4. **Transformer-based modeling**
5. **Emotion classification & statistical evaluation**

---

## Repository Structure

- `vr_transformer/` — Transformer architectures for VR interaction sequences  
- `config_*.py` — Experiment and model configuration files  
- `preprocess_vr_data.py` — Preprocess raw VR logs into flattened CSV representations  
- `main_emotionRecognition.py` — End-to-end training and evaluation pipeline  
- `train_pretrain.py` — Model pretraining stage  
- `train_finetune.py` — Model finetuning stage  
- `evaluate_only.py` — Model evaluation and inference  
- `utils_labels.py` — Emotion label handling and mapping utilities  
- `wilcoxon.py` — Statistical significance testing (Wilcoxon signed-rank test)




---

## Data

This repository includes **processed CSV files** used in experiments:

- `chunked_vr_data.csv` – VR interaction sequences
- `chunked_with_textProj.csv` – VR + text-projected features
- `featureVector_original.csv` – Aggregated feature representation

> ⚠️ Raw VR logs are not included for privacy and ethical reasons.

---

## Requirements

- Python ≥ 3.9
- PyTorch
- NumPy
- Pandas
- scikit-learn

---

## Reproducibility Notes

- Fixed temporal windowing and chunk size
- Explicit configuration files
- Deterministic data preprocessing
- Clear separation of preprocessing, training, and evaluation

This repository is intended for **research use**, not as a production system.

---

## Citation

If you use this code in academic work, please cite it:

```bibtex
@software{Alaei_VRInteraction_2026,
  title  = {VR Interaction as a Language-Like Sequence},
  author = {Alaei, Sharifeh},
  year   = {2026},
  url    = {https://github.com/sharifaaaa/vr-interaction-language-model}
}


