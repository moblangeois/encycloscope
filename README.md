# EncyclopedIA_encycloscope

## Vue d'ensemble

EncyclopedIA_encycloscope est un projet de recherche en intelligence artificielle qui développe un moteur de recherche conceptuel avancé basé sur l'**Encyclopédie de Diderot et d'Alembert**. Le projet utilise des techniques modernes de traitement du langage naturel (NLP) et d'apprentissage automatique pour créer un système de recherche sémantique intelligent.

## Objectifs du projet

- **Création d'un moteur de recherche conceptuel** : Développement d'un système de recherche avancé qui comprend les concepts et les relations sémantiques dans l'Encyclopédie
- **Modélisation linguistique spécialisée** : Entraînement de modèles Word2Vec et CamemBERT spécifiquement adaptés au vocabulaire et au style de l'époque des Lumières
- **Interface de recherche intelligente** : Développement d'un pipeline de recherche utilisant des embeddings vectoriels et des techniques de RAG (Retrieval-Augmented Generation)

## Architecture du projet

### Notebooks principaux

1. **`1_creation_modele_word2vec.ipynb`**
   - Génération d'un modèle Word2Vec spécialisé pour l'Encyclopédie
   - Préparation et nettoyage des corpus textuels
   - Validation des relations sémantiques découvertes

2. **`2_creation_modele_camembert.ipynb`**
   - Fine-tuning d'un modèle CamemBERT sur les données de l'Encyclopédie
   - Création de paires d'entraînement de haute qualité
   - Évaluation et optimisation du modèle

3. **`3_test_moteur_recherche_conceptuel.ipynb`**
   - Tests et évaluation du moteur de recherche conceptuel
   - Analyse des performances de recherche
   - Validation des résultats

### Scripts principaux

- **`encycloscope1.py`** : Implementation du pipeline de recherche avec support OpenAI et Nomic
- **`encycloscope3.py`** : Version avancée du moteur de recherche
- **`config.py`** : Configuration centralisée des paramètres d'entraînement et de recherche
- **`utils.py`** : Fonctions utilitaires communes

## Structure des dossiers

```
EncyclopedIA_encycloscope/
├── input/                          # Données d'entrée
├── output/                         # Résultats et modèles générés
│   ├── embeddings/                 # Embeddings vectoriels
│   ├── models/                     # Modèles entraînés
│   │   └── encyclopedia_camembert_aligned/
│   └── word2vec/                   # Modèles Word2Vec
├── 1_creation_modele_word2vec.ipynb
├── 2_creation_modele_camembert.ipynb
├── 3_test_moteur_recherche_conceptuel.ipynb
├── encycloscope1.py
├── encycloscope3.py
├── config.py
├── utils.py
└── requirements.txt
```

## Installation

### Prérequis

- Python 3.8+
- CUDA compatible GPU (recommandé pour l'entraînement des modèles)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Dépendances principales

- **Apprentissage automatique** : PyTorch, Transformers, Sentence-Transformers, Scikit-learn
- **NLP spécialisé** : Gensim, spaCy, CamemBERT
- **Recherche vectorielle** : FAISS, LlamaIndex
- **APIs** : OpenAI, Nomic
- **Analyse de données** : NumPy, Pandas, Matplotlib, Seaborn

## Configuration

Le fichier `config.py` contient tous les paramètres configurables :

### Paramètres d'entraînement
```python
TRAINING_PARAMS = {
    "epochs": 3,
    "batch_size": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_pairs": 50000,
    "model_output_name": "encyclopedia_camembert_aligned"
}
```

### Paramètres de recherche
```python
SEARCH_PARAMS = {
    "top_k": 10,
    "max_top_k": 50,
    "embedding_batch_size": 32,
    "search_algorithm": "cosine"
}
```

## Utilisation

### 1. Préparation des données

Placez vos données de l'Encyclopédie dans le dossier `input/` au format CSV avec les colonnes requises :
- `articleID` : Identifiant unique de l'article
- `vedette` : Titre de l'article
- `contenu_clean` : Contenu nettoyé de l'article
- `domainesEnccre` : Domaine thématique
- `URLEnccre` : URL de référence

### 2. Entraînement des modèles

Exécutez les notebooks dans l'ordre :

1. **Word2Vec** : `1_creation_modele_word2vec.ipynb`
2. **CamemBERT** : `2_creation_modele_camembert.ipynb`
3. **Tests** : `3_test_moteur_recherche_conceptuel.ipynb`

---

*Ce projet fait partie d'une initiative de recherche visant à rendre accessible et explorable l'héritage intellectuel des Lumières à travers les technologies modernes d'intelligence artificielle.*
