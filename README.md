# VR Interaction as a Language-Like Sequence

This repository contains code for modeling embodied interaction in Virtual Reality (VR) as
language-like sequential data, using Transformer-based architectures.

## Project Motivation

Human behavior in immersive environments unfolds over time through interaction.
This project explores whether embodied VR interaction logs (gaze, motion, actions)
can be treated as structured sequences, analogous to natural language, for downstream
tasks such as emotion inference and cognitive state modeling.

## Repository Structure

vr_transformer/                  # Transformer models for VR interaction
config_*.py                      # Experiment configurations
preprocess_vr_Data.py	           #preprocess raw json file and convert it a flattened CSV file
main_emotionRecognition.py       # main pipeline
train_pretrain.py                # Model training 
train_finetune.py                # Model finetuning
evaluate_only.py                 #Model evaluation
util_labels.py                   #manging emotion labels
wilcoxon.py		                   #Statistical tests


