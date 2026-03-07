"""
Script pour reconstruire l'index FAISS avec d'AlemBERT
Utilise le CSV encyclopedia_blocks.csv existant

Usage:
    python rebuild_faiss_index.py

Output:
    - data/encyclopedia_dalembert.faiss  (index FAISS)
    - data/encyclopedia_dalembert_data.pkl (métadonnées)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from dalembert_encoder import DAlemBERTEncoder


def main():
    print("=" * 80)
    print("RECONSTRUCTION DE L'INDEX FAISS AVEC D'ALEMBERT")
    print("=" * 80)

    # Configuration
    CSV_PATH = 'output/encyclopedia_blocks.csv'
    TEXT_COLUMN = 'block_text'
    SAVE_DIR = 'data'
    SAVE_NAME = 'encyclopedia_dalembert'
    BATCH_SIZE = 16  # Ajuster selon la VRAM disponible

    # Créer le dossier data si nécessaire
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Charger les données
    print(f"\n1. Chargement des données depuis {CSV_PATH}...")

    if not os.path.exists(CSV_PATH):
        print(f"   ERREUR: Fichier non trouvé: {CSV_PATH}")
        print(f"   Chemin absolu: {os.path.abspath(CSV_PATH)}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"   {len(df)} blocs chargés")
    print(f"   Colonnes: {df.columns.tolist()}")

    # Extraire les textes
    texts = df[TEXT_COLUMN].fillna('').tolist()
    print(f"   Textes extraits de la colonne '{TEXT_COLUMN}'")

    # Extraire les métadonnées (toutes les colonnes sauf le texte)
    metadata_columns = [col for col in df.columns if col != TEXT_COLUMN]
    metadata = df[metadata_columns].to_dict('index')
    print(f"   Métadonnées: {metadata_columns}")

    # 2. Charger d'AlemBERT
    print("\n2. Chargement de d'AlemBERT...")
    encoder = DAlemBERTEncoder('pjox/dalembert', pooling='mean')

    # 3. Encoder les textes
    print(f"\n3. Encodage de {len(texts)} blocs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Nombre de batches: {(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"\n   Embeddings shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")

    # Vérifier la normalisation
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # 4. Créer l'index FAISS
    print("\n4. Création de l'index FAISS...")
    dimension = embeddings.shape[1]
    print(f"   Dimension: {dimension}")

    # IndexFlatIP = Inner Product (équivalent cosine si vecteurs normalisés)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    print(f"   {index.ntotal} vecteurs ajoutés à l'index")

    # 5. Tester l'index
    print("\n5. Test de l'index...")
    test_query = "économie politique"
    print(f"   Requête test: '{test_query}'")

    query_vec = encoder.encode([test_query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec.astype(np.float32), k=5)

    print(f"\n   Top 5 résultats:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        title = df.iloc[idx].get('article_head', 'Inconnu')
        text_preview = texts[idx][:80].replace('\n', ' ')
        print(f"   [{i+1}] Score: {score:.4f} | {title}")
        print(f"       {text_preview}...")

    # 6. Sauvegarder
    print(f"\n6. Sauvegarde de l'index...")

    faiss_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}.faiss")
    data_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}_data.pkl")

    # Sauvegarder l'index FAISS
    faiss.write_index(index, faiss_path)
    print(f"   Index FAISS: {faiss_path}")

    # Sauvegarder les données associées
    with open(data_path, "wb") as f:
        pickle.dump({
            'texts': texts,
            'metadata': metadata,
            'columns': metadata_columns,
            'csv_path': CSV_PATH
        }, f)
    print(f"   Données: {data_path}")

    print("\n" + "=" * 80)
    print("RECONSTRUCTION TERMINÉE")
    print("=" * 80)
    print(f"\nFichiers créés:")
    print(f"  - {faiss_path}")
    print(f"  - {data_path}")
    print(f"\nPour utiliser dans encycloscope, mettre à jour les valves:")
    print(f"  INDEX_PATH: '{faiss_path}'")
    print(f"  DATA_PATH: '{data_path}'")


if __name__ == "__main__":
    main()
