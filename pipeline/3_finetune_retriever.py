#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Étape 3: Fine-tuning du retriever avec MNRL
============================================

Fine-tune le modèle TSDAE avec les données GPL en utilisant
MultipleNegativesRankingLoss (InfoNCE).

MNRL est le standard pour le fine-tuning de retrieval dense.
Elle utilise tous les positifs du batch comme négatifs implicites
+ les hard negatives explicites de GPL.

Usage:
    python 3_finetune_retriever.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

# Config
from config import (
    TSDAE_MODEL_DIR,
    GPL_DATA_DIR,
    FINAL_MODEL_DIR,
    BLOCKS_CSV,
    FINETUNE_CONFIG,
    INDEX_CONFIG,
    HARDWARE_CONFIG,
    OUTPUT_DIR,
    BASE_MODEL,
)


def load_gpl_data(gpl_path: Path) -> list[dict]:
    """Charge les données GPL."""
    print(f"Chargement données GPL: {gpl_path}")
    with open(gpl_path, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} triplets")
    return data


def create_training_examples(gpl_data: list[dict], config: dict) -> list[InputExample]:
    """
    Convertit les données GPL en InputExamples pour MNRL.

    Pour MNRL avec hard negatives, on crée des paires (anchor, positive)
    et les hard negatives sont ajoutées via la loss.
    """
    print(f"\nCréation des exemples d'entraînement...")

    examples = []

    for item in gpl_data:
        query = item["query"]
        positive = item["positive"]["passage"]

        # Paire principale (query, positive)
        examples.append(InputExample(texts=[query, positive]))

        # Ajouter des paires avec hard negatives pour plus de contraste
        # (optionnel, MNRL utilise aussi les in-batch negatives)
        for neg in item["negatives"][: config["hard_negatives_per_query"]]:
            # On peut créer des triplets (anchor, positive, negative)
            # mais MNRL standard utilise juste des paires
            pass

    print(f"  {len(examples)} exemples créés")
    return examples


def train_retriever(
    model_path: Path,
    examples: list[InputExample],
    output_path: Path,
    config: dict,
):
    """
    Fine-tune le retriever avec MNRL.

    Args:
        model_path: Chemin vers le modèle TSDAE (ou base si TSDAE pas dispo)
        examples: Exemples d'entraînement
        output_path: Dossier de sortie
        config: Paramètres d'entraînement
    """
    print(f"\n{'='*60}")
    print("Fine-tuning avec MultipleNegativesRankingLoss")
    print(f"{'='*60}")

    # Charger le modèle
    # Note: TSDAE skip car incompatible avec sentence-camembert-large
    # On utilise directement le modèle de base pré-entraîné pour embeddings
    print(f"Chargement du modèle de base: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    print(f"Exemples: {len(examples)}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch: {config['batch_size'] * config['gradient_accumulation_steps']}")

    # DataLoader
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=config["batch_size"],
    )

    # Loss MNRL (InfoNCE)
    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=config["loss_scale"],
    )

    # Warmup
    total_steps = len(train_dataloader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    print(f"\nTotal steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

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
        checkpoint_save_steps=len(train_dataloader),  # Save each epoch
    )

    print(f"\nModèle sauvegardé: {output_path}")
    return model


def create_search_index(
    model: SentenceTransformer,
    blocks_path: Path,
    output_dir: Path,
    config: dict,
):
    """
    Crée l'index de recherche final avec le modèle fine-tuné.
    """
    print(f"\n{'='*60}")
    print("Création de l'index de recherche")
    print(f"{'='*60}")

    # Charger les blocs
    df = pd.read_csv(blocks_path)
    print(f"Blocs: {len(df)}")

    # Trouver la colonne texte
    text_column = None
    for col in ["block_text", "text", "contenu_clean"]:
        if col in df.columns:
            text_column = col
            break

    if not text_column:
        raise ValueError(f"Colonne texte non trouvée: {df.columns.tolist()}")

    texts = df[text_column].fillna("").tolist()

    # Générer embeddings
    print(f"Génération des embeddings (batch={config['embedding_batch_size']})...")
    embeddings = model.encode(
        texts,
        batch_size=config["embedding_batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"  Shape: {embeddings.shape}")

    # Créer index NearestNeighbors
    print(f"Création de l'index (k={config['n_neighbors']}, metric={config['metric']})...")
    index = NearestNeighbors(
        n_neighbors=config["n_neighbors"],
        algorithm="auto",
        metric=config["metric"],
    )
    index.fit(embeddings)

    # Sauvegarder
    embeddings_file = output_dir / "encyclopedia_embeddings.npy"
    index_file = output_dir / "encyclopedia_index.pkl"
    blocks_file = output_dir / "encyclopedia_blocks.csv"

    np.save(embeddings_file, embeddings)
    with open(index_file, "wb") as f:
        pickle.dump(index, f)
    df.to_csv(blocks_file, index=False)

    print(f"\nFichiers générés:")
    print(f"  - {embeddings_file}")
    print(f"  - {index_file}")
    print(f"  - {blocks_file}")

    return embeddings, index


def main():
    print("=" * 60)
    print("ÉTAPE 3: Fine-tuning du retriever")
    print("=" * 60)

    # Vérifier GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Mode CPU")

    # Créer dossiers
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Charger données GPL
    gpl_file = GPL_DATA_DIR / "gpl_training_data.pkl"
    if not gpl_file.exists():
        print(f"ERREUR: {gpl_file} non trouvé")
        print("Exécutez d'abord: python 2_gpl_generation.py")
        sys.exit(1)

    gpl_data = load_gpl_data(gpl_file)

    # Créer exemples
    examples = create_training_examples(gpl_data, FINETUNE_CONFIG)

    if len(examples) < 100:
        print(f"ATTENTION: Seulement {len(examples)} exemples, résultats potentiellement mauvais")

    # Fine-tuning
    model = train_retriever(
        model_path=TSDAE_MODEL_DIR,
        examples=examples,
        output_path=FINAL_MODEL_DIR,
        config=FINETUNE_CONFIG,
    )

    # Créer index de recherche
    if BLOCKS_CSV.exists():
        create_search_index(
            model=model,
            blocks_path=BLOCKS_CSV,
            output_dir=FINAL_MODEL_DIR,
            config=INDEX_CONFIG,
        )

    # Résumé
    print(f"\n{'='*60}")
    print("PIPELINE TERMINÉ!")
    print(f"{'='*60}")
    print(f"Modèle final: {FINAL_MODEL_DIR}")
    print(f"\nPour utiliser le nouveau modèle dans Encycloscope:")
    print(f"  1. Copier {FINAL_MODEL_DIR} vers le serveur")
    print(f"  2. Mettre à jour MODEL_PATH dans encycloscope_version_3_0.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
