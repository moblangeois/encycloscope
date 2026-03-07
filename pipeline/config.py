"""
Configuration centralisée du pipeline de fine-tuning
=====================================================

Pipeline recommandé par deep research :
1. TSDAE - adaptation domaine non supervisée
2. GPL - génération queries synthétiques + pseudo-labels
3. Fine-tuning MNRL avec hard negatives
"""

from pathlib import Path

# =============================================================================
# CHEMINS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PIPELINE_DIR = BASE_DIR / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"

# Données sources
BLOCKS_CSV = BASE_DIR / "output" / "encyclopedia_blocks.csv"
WORD2VEC_MODEL = BASE_DIR.parent / "word2vec" / "model" / "enccreModelFromChunk"

# Sorties intermédiaires
TSDAE_MODEL_DIR = OUTPUT_DIR / "1_tsdae"
GPL_DATA_DIR = OUTPUT_DIR / "2_gpl"
FINAL_MODEL_DIR = OUTPUT_DIR / "3_final_model"

# =============================================================================
# MODÈLES DE BASE
# =============================================================================

# Choix du modèle de base pour le retriever
# Options recommandées :
# - "sentence-camembert-large" : bon pour le français, pré-entraîné embeddings
# - "BAAI/bge-m3" : état de l'art multilingue, 8K context
# - "dangvantuan/sentence-camembert-large" : variante sentence-transformers
BASE_MODEL = "dangvantuan/sentence-camembert-large"

# Modèle pour générer les queries (GPL)
# Options : "qwen2.5:14b-instruct", "qwen3:8b", "gemma3:12b"
# Pour local via Ollama
QUERY_GENERATOR_MODEL = "qwen2.5:14b-instruct"  # Via Ollama local
QUERY_GENERATOR_USE_API = False  # True pour OpenAI/Anthropic

# Cross-encoder pour pseudo-labeling (GPL)
CROSS_ENCODER_MODEL = "antoinelouis/crossencoder-camembert-base-mmarcoFR"

# =============================================================================
# PARAMÈTRES TSDAE
# =============================================================================

TSDAE_CONFIG = {
    "epochs": 1,  # 1 epoch suffit généralement
    "batch_size": 8,  # Ajuster selon VRAM
    "deletion_ratio": 0.6,  # Proportion de mots supprimés
    "learning_rate": 3e-5,
    "max_seq_length": 256,
    "warmup_steps": 1000,
}

# =============================================================================
# PARAMÈTRES GPL
# =============================================================================

GPL_CONFIG = {
    # Génération de queries
    "queries_per_passage": 1,  # Réduit de 3 à 1 pour vitesse
    "max_passages": 10000,  # Échantillon (None = tous)
    "max_passage_length": 512,

    # Negative mining
    "negatives_per_query": 50,  # Hard negatives à récupérer
    "negative_threshold": 0.6,  # Filtrage: si score_neg > 0.6*score_pos, exclure

    # Pseudo-labeling
    "use_cross_encoder": True,  # Utiliser cross-encoder pour scores

    # Filtrage qualité
    "filter_ratio": 0.3,  # Garder top 30% des queries générées
}

# =============================================================================
# PARAMÈTRES FINE-TUNING
# =============================================================================

FINETUNE_CONFIG = {
    "epochs": 10,
    "batch_size": 16,
    "gradient_accumulation_steps": 4,  # Effective batch = 64
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_seq_length": 256,

    # Loss
    "loss_scale": 20.0,  # Temperature pour InfoNCE
    "use_mnrl": True,  # MultipleNegativesRankingLoss

    # Hard negatives
    "hard_negatives_per_query": 5,

    # Gradient caching pour gros batch effectif
    "use_grad_cache": True,
    "mini_batch_size": 32,
}

# =============================================================================
# PARAMÈTRES INDEX FINAL
# =============================================================================

INDEX_CONFIG = {
    "n_neighbors": 100,  # Top-K pour retrieval initial
    "metric": "cosine",
    "embedding_batch_size": 32,
}

# =============================================================================
# HARDWARE
# =============================================================================

HARDWARE_CONFIG = {
    "device": "cuda",  # "cuda" ou "cpu"
    "fp16": True,  # Mixed precision
    "gradient_checkpointing": True,  # Économise VRAM
}
