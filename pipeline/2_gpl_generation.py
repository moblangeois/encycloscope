#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Étape 2: GPL - Generative Pseudo Labeling
==========================================

Génère des données d'entraînement synthétiques de haute qualité.

Pipeline GPL:
1. Génération de queries : LLM génère des questions pour chaque passage
2. Negative mining : Word2Vec d'Aurélia trouve des hard negatives
3. Pseudo-labeling : Cross-encoder assigne des scores continus

Innovation: On utilise Word2Vec d'Aurélia pour le negative mining,
ce qui exploite sa connaissance sémantique du corpus.

Référence: https://arxiv.org/abs/2112.07577
Gain attendu: +7-9 points nDCG@10 vs baseline

Usage:
    python 2_gpl_generation.py
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

# Config
from config import (
    BLOCKS_CSV,
    WORD2VEC_MODEL,
    TSDAE_MODEL_DIR,
    GPL_DATA_DIR,
    GPL_CONFIG,
    CROSS_ENCODER_MODEL,
    QUERY_GENERATOR_MODEL,
    QUERY_GENERATOR_USE_API,
)


def load_blocks(blocks_path: Path) -> pd.DataFrame:
    """Charge les blocs de texte."""
    print(f"Chargement des blocs: {blocks_path}")
    df = pd.read_csv(blocks_path)
    print(f"  {len(df)} blocs")
    return df


def load_word2vec(model_path: Path) -> Word2Vec:
    """Charge le modèle Word2Vec d'Aurélia."""
    print(f"Chargement Word2Vec: {model_path}")
    model = Word2Vec.load(str(model_path))
    print(f"  Vocabulaire: {len(model.wv)} mots")
    return model


def generate_query_ollama(passage: str, model: str = "qwen2.5:14b-instruct") -> Optional[str]:
    """Génère une question avec Ollama (local)."""
    try:
        import ollama

        prompt = f"""Voici un extrait de l'Encyclopédie de Diderot et d'Alembert (1751-1772).
Génère une question qu'un chercheur ou étudiant pourrait poser pour trouver ce passage.
La question doit être en français et naturelle.

Passage: {passage[:1000]}

Question:"""

        response = ollama.generate(model=model, prompt=prompt)
        query = response["response"].strip()

        # Nettoyer la réponse
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        if query.startswith("Question:"):
            query = query[9:].strip()

        return query if len(query) > 10 else None

    except Exception as e:
        print(f"  Erreur Ollama: {e}")
        return None


def generate_queries_for_passages(
    df: pd.DataFrame,
    queries_per_passage: int = 3,
    text_column: str = "block_text",
) -> list[dict]:
    """
    Génère des questions pour chaque passage.

    Returns:
        Liste de dicts {passage_id, passage, query}
    """
    print(f"\nGénération de queries ({queries_per_passage} par passage)...")

    # Trouver la colonne texte
    for col in [text_column, "text", "contenu_clean"]:
        if col in df.columns:
            text_column = col
            break

    generated = []
    total = len(df)

    for idx, row in tqdm(df.iterrows(), total=total, desc="Génération queries"):
        passage = str(row[text_column])

        if len(passage) < 100:
            continue

        for _ in range(queries_per_passage):
            query = generate_query_ollama(passage)
            if query:
                generated.append({
                    "passage_id": idx,
                    "passage": passage,
                    "query": query,
                })

        # Progress
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{total} passages traités, {len(generated)} queries générées")

    print(f"Total: {len(generated)} queries générées")
    return generated


def find_hard_negatives_w2v(
    query: str,
    positive_passage: str,
    df: pd.DataFrame,
    w2v_model: Word2Vec,
    n_negatives: int = 50,
    text_column: str = "block_text",
) -> list[dict]:
    """
    Trouve des hard negatives en utilisant Word2Vec.

    Stratégie: Chercher des passages qui partagent du vocabulaire
    similaire (selon W2V) mais qui ne sont pas le passage positif.
    Ce sont les "hard" negatives - sémantiquement proches mais pas pertinents.
    """
    # Extraire les mots de la query
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    query_words = [w for w in query_words if w in w2v_model.wv]

    if not query_words:
        return []

    # Trouver des mots similaires selon W2V
    similar_words = set()
    for word in query_words[:5]:  # Top 5 mots de la query
        try:
            for similar_word, _ in w2v_model.wv.most_similar(word, topn=10):
                similar_words.add(similar_word)
        except KeyError:
            continue

    if not similar_words:
        return []

    # Trouver des passages contenant ces mots similaires
    negatives = []
    for col in [text_column, "text", "contenu_clean"]:
        if col in df.columns:
            text_column = col
            break

    for idx, row in df.iterrows():
        passage = str(row[text_column]).lower()

        # Skip le passage positif
        if passage[:100] == positive_passage[:100].lower():
            continue

        # Compter les mots similaires présents
        matches = sum(1 for w in similar_words if w in passage)

        if matches > 0:
            negatives.append({
                "passage_id": idx,
                "passage": str(row[text_column]),
                "w2v_score": matches / len(similar_words),
            })

    # Trier par score W2V (plus de matches = plus "hard")
    negatives.sort(key=lambda x: x["w2v_score"], reverse=True)

    return negatives[:n_negatives]


def score_with_cross_encoder(
    queries_data: list[dict],
    cross_encoder: CrossEncoder,
    df: pd.DataFrame,
    w2v_model: Word2Vec,
    config: dict,
) -> list[dict]:
    """
    Score les paires (query, passage) avec le cross-encoder
    et ajoute les hard negatives.

    Returns:
        Liste de triplets {query, positive, negatives, scores}
    """
    print(f"\nScoring avec cross-encoder et negative mining...")

    training_data = []

    for item in tqdm(queries_data, desc="Cross-encoder scoring"):
        query = item["query"]
        positive_passage = item["passage"]

        # Score du positif
        pos_score = cross_encoder.predict([(query, positive_passage)])[0]

        # Trouver hard negatives avec W2V
        negatives = find_hard_negatives_w2v(
            query=query,
            positive_passage=positive_passage,
            df=df,
            w2v_model=w2v_model,
            n_negatives=config["negatives_per_query"],
        )

        if not negatives:
            continue

        # Scorer les negatives
        neg_passages = [n["passage"] for n in negatives]
        neg_scores = cross_encoder.predict([(query, p) for p in neg_passages])

        # Filtrer les faux négatifs (score trop élevé)
        threshold = config["negative_threshold"] * pos_score
        valid_negatives = []
        for neg, score in zip(negatives, neg_scores):
            if score < threshold:
                valid_negatives.append({
                    "passage": neg["passage"],
                    "score": float(score),
                })

        if not valid_negatives:
            continue

        training_data.append({
            "query": query,
            "positive": {
                "passage": positive_passage,
                "score": float(pos_score),
            },
            "negatives": valid_negatives[:config["negatives_per_query"]],
        })

    print(f"Total: {len(training_data)} triplets d'entraînement")
    return training_data


def main():
    print("=" * 60)
    print("ÉTAPE 2: GPL - Generative Pseudo Labeling")
    print("=" * 60)

    # Créer dossiers
    GPL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Vérifier fichiers
    if not BLOCKS_CSV.exists():
        print(f"ERREUR: {BLOCKS_CSV} non trouvé")
        sys.exit(1)

    if not WORD2VEC_MODEL.exists():
        print(f"ERREUR: {WORD2VEC_MODEL} non trouvé")
        print("Le modèle Word2Vec d'Aurélia est requis pour le negative mining.")
        sys.exit(1)

    # Charger données
    df = load_blocks(BLOCKS_CSV)
    w2v_model = load_word2vec(WORD2VEC_MODEL)

    # Échantillonner si max_passages défini
    max_passages = GPL_CONFIG.get("max_passages")
    if max_passages and len(df) > max_passages:
        df = df.sample(n=max_passages, random_state=42)
        print(f"  Échantillon: {len(df)} blocs (max_passages={max_passages})")

    # Étape 1: Génération de queries
    queries_cache = GPL_DATA_DIR / "generated_queries.json"

    if queries_cache.exists():
        print(f"\nChargement queries depuis cache: {queries_cache}")
        with open(queries_cache, "r", encoding="utf-8") as f:
            queries_data = json.load(f)
    else:
        queries_data = generate_queries_for_passages(
            df,
            queries_per_passage=GPL_CONFIG["queries_per_passage"],
        )
        # Sauvegarder
        with open(queries_cache, "w", encoding="utf-8") as f:
            json.dump(queries_data, f, ensure_ascii=False, indent=2)
        print(f"Queries sauvegardées: {queries_cache}")

    # Filtrer top queries (qualité)
    n_keep = int(len(queries_data) * (1 - GPL_CONFIG["filter_ratio"]))
    queries_data = queries_data[:n_keep]
    print(f"Après filtrage: {len(queries_data)} queries")

    # Étape 2: Cross-encoder scoring + negative mining
    print(f"\nChargement cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    training_data = score_with_cross_encoder(
        queries_data=queries_data,
        cross_encoder=cross_encoder,
        df=df,
        w2v_model=w2v_model,
        config=GPL_CONFIG,
    )

    # Sauvegarder
    output_file = GPL_DATA_DIR / "gpl_training_data.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(training_data, f)
    print(f"\nDonnées GPL sauvegardées: {output_file}")

    # Stats
    print(f"\n{'='*60}")
    print("RÉSUMÉ GPL")
    print(f"{'='*60}")
    print(f"Queries générées: {len(queries_data)}")
    print(f"Triplets d'entraînement: {len(training_data)}")
    if training_data:
        avg_negs = np.mean([len(t["negatives"]) for t in training_data])
        print(f"Negatives par query (moy): {avg_negs:.1f}")
    print(f"\nProchaine étape: python 3_finetune_retriever.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
