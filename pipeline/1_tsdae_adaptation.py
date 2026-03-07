#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Étape 1: TSDAE - Adaptation de domaine non supervisée
=====================================================

TSDAE (Transformer-based Denoising Auto-Encoder) adapte le modèle
au vocabulaire et style de l'Encyclopédie sans labels.

Principe:
- Supprime ~60% des mots d'un passage
- L'encodeur produit un embedding
- Le décodeur reconstruit le texte original
- Le modèle apprend des représentations denses du corpus

Référence: https://arxiv.org/abs/2104.06979
Atteint ~93% des performances supervisées en mode unsupervised.

Usage:
    python 1_tsdae_adaptation.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

# Config
from config import (
    BASE_MODEL,
    BLOCKS_CSV,
    TSDAE_MODEL_DIR,
    TSDAE_CONFIG,
    HARDWARE_CONFIG,
    OUTPUT_DIR,
)


def load_corpus(blocks_path: Path, text_column: str = "block_text") -> list[str]:
    """Charge le corpus de textes pour TSDAE."""
    print(f"Chargement du corpus: {blocks_path}")

    df = pd.read_csv(blocks_path)
    print(f"  {len(df)} blocs chargés")

    # Colonnes possibles pour le texte
    for col in [text_column, "text", "contenu_clean"]:
        if col in df.columns:
            texts = df[col].dropna().tolist()
            print(f"  Colonne utilisée: {col}")
            print(f"  {len(texts)} textes valides")
            return texts

    raise ValueError(f"Colonne texte non trouvée. Colonnes: {df.columns.tolist()}")


def train_tsdae(
    texts: list[str],
    model_name: str,
    output_path: Path,
    config: dict,
):
    """
    Entraîne TSDAE sur le corpus.

    Args:
        texts: Liste de passages du corpus
        model_name: Modèle de base (ex: sentence-camembert-large)
        output_path: Dossier de sortie
        config: Paramètres TSDAE
    """
    print(f"\n{'='*60}")
    print("TSDAE - Adaptation de domaine")
    print(f"{'='*60}")
    print(f"Modèle de base: {model_name}")
    print(f"Corpus: {len(texts)} passages")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Deletion ratio: {config['deletion_ratio']}")

    # Charger le modèle
    print(f"\nChargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)

    if HARDWARE_CONFIG["device"] == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  CPU mode")

    # Dataset TSDAE
    print(f"\nPréparation du dataset TSDAE...")
    train_dataset = DenoisingAutoEncoderDataset(texts)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    print(f"  {len(train_dataloader)} batches")

    # Loss TSDAE
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=True,
    )

    # Warmup
    warmup_steps = min(config["warmup_steps"], len(train_dataloader))
    total_steps = len(train_dataloader) * config["epochs"]

    print(f"\nDébut de l'entraînement...")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")

    # Entraînement
    output_path.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config["epochs"],
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config["learning_rate"]},
        output_path=str(output_path),
        show_progress_bar=True,
        use_amp=HARDWARE_CONFIG["fp16"],
    )

    print(f"\nModèle TSDAE sauvegardé: {output_path}")
    return model


def main():
    print("=" * 60)
    print("ÉTAPE 1: TSDAE - Adaptation de domaine")
    print("=" * 60)

    # Créer dossiers
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Vérifier les fichiers
    if not BLOCKS_CSV.exists():
        print(f"ERREUR: Fichier non trouvé: {BLOCKS_CSV}")
        print("Exécutez d'abord le notebook de préparation des données.")
        sys.exit(1)

    # Charger corpus
    texts = load_corpus(BLOCKS_CSV)

    # Limiter pour test rapide (enlever en production)
    # texts = texts[:1000]
    # print(f"  [TEST] Limité à {len(texts)} textes")

    # Entraîner TSDAE
    model = train_tsdae(
        texts=texts,
        model_name=BASE_MODEL,
        output_path=TSDAE_MODEL_DIR,
        config=TSDAE_CONFIG,
    )

    print("\n" + "=" * 60)
    print("TSDAE terminé!")
    print(f"Modèle adapté: {TSDAE_MODEL_DIR}")
    print("Prochaine étape: python 2_gpl_generation.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
