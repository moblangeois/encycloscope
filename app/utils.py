"""
Fonctions utilitaires pour le traitement des données de l'Encyclopédie 
et l'entraînement du modèle d'embeddings avec approche question-passage alignée.
"""

import gc
import os
import pickle
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tiktoken
from sentence_transformers import InputExample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from config import (
    MODEL_NAME, TRAINING_PARAMS, GENERATION_PARAMS, SEARCH_PARAMS,
    INPUT_DIR, OUTPUT_DIR, DATA_FILE, MODEL_DIR, EMBEDDINGS_DIR,
    OUTPUT_FILES, COLUMN_MAPPING, URL_TEMPLATE, RAG_COMPATIBILITY
)

# Supprimer les avertissements pour une sortie plus propre
warnings.filterwarnings("ignore")

# Configuration des constantes depuis config.py
DEFAULT_MAX_TOKENS = 512  # Longueur maximale des séquences
DEFAULT_MIN_TOKENS = 50
BATCH_SIZE = TRAINING_PARAMS["batch_size"]


def clean_text_for_bert(text: str) -> str:
    """
    Nettoie le texte pour le fine-tuning d'un modèle CamemBERT
    tout en préservant la structure grammaticale et syntaxique.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    text = text.replace('\r\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Extraire le contenu principal après le désignant entre parenthèses
    match = re.match(r'^(.*?)\((.*?)\)(.*?)$', text)
    
    if match:
        vedette = match.group(1).strip()
        designant = match.group(2).strip()
        contenu_principal = match.group(3).strip()
        text = f"{vedette} ({designant}) {contenu_principal}"
    
    return text


def get_token_count(text: str, model: str = "cl100k_base") -> int:
    """Compte le nombre exact de tokens dans un texte selon le modèle spécifié."""
    encoder = tiktoken.get_encoding(model)
    tokens = encoder.encode(text)
    return len(tokens)


def split_text_by_tokens(text: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = 0) -> List[str]:
    """
    Découpe un texte en blocs respectant strictement la limite de tokens,
    avec possibilité de chevauchement contrôlé.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    blocks = []
    i = 0
    while i < len(tokens):
        block_tokens = tokens[i:i + max_tokens]
        block_text = encoder.decode(block_tokens)
        blocks.append(block_text)
        i += max_tokens - overlap
    
    return blocks


def is_redirect_article(text: str) -> bool:
    """Détecte si un article est principalement une redirection vers un autre article."""
    text_lower = text.lower()
    
    redirect_patterns = [
        r'voyez\s+\w+',
        r'voy\.\s+\w+',
        r'v\.\s+\w+',
        r'voyez\s+l\'article',
        r'se\s+rapporter\s+à'
    ]
    
    if len(text.split()) < 50:
        for pattern in redirect_patterns:
            if re.search(pattern, text_lower):
                return True
    
    return False


def is_informative_article(text: str, min_words: int = 20, min_unique_ratio: float = 0.4) -> bool:
    """
    Détermine si un article contient suffisamment d'information.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < min_words:
        return False
    
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words)
    
    return unique_ratio >= min_unique_ratio


def prepare_article_blocks(df: pd.DataFrame, max_tokens: int = DEFAULT_MAX_TOKENS, 
                          min_block_tokens: int = DEFAULT_MIN_TOKENS) -> pd.DataFrame:
    """
    Prépare les blocs d'articles pour l'indexation avec colonnes standardisées pour la compatibilité RAG.
    """
    doc_blocks: List[Dict[str, Any]] = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Préparation des blocs"):
        article_text = str(row.get('contenu_clean', ''))
        article_id = row.get('articleID', f'article_{idx}')
        
        # Ignorer les articles trop courts ou peu informatifs
        if len(article_text.split()) < min_block_tokens:
            continue
            
        # Découper le texte en blocs de tokens
        blocks = split_text_by_tokens(article_text, max_tokens=max_tokens)
        
        # Ajouter chaque bloc valide avec colonnes standardisées
        for block_idx, block in enumerate(blocks):
            if get_token_count(block) >= min_block_tokens:
                # Créer un ID unique pour le bloc
                block_id = f"{article_id}_block_{block_idx}"
                
                # Colonnes standardisées pour la compatibilité RAG
                doc_blocks.append({
                    # Colonnes requises par le script RAG
                    'id': block_id,
                    'text': block,
                    'title': row.get('vedette', 'Titre inconnu'),
                    'domain': row.get('domainesEnccre', 'Non classifié'),
                    'url': URL_TEMPLATE.format(article_id=article_id),
                    
                    # Colonnes héritées pour compatibilité
                    'articleID': article_id,
                    'vedette': row.get('vedette', 'Titre inconnu'),
                    'domainesEnccre': row.get('domainesEnccre', 'non_classifié'),
                    'designants': row.get('designants', 'non_classifié'),
                    'auteurs': row.get('auteurs', 'Inconnu'),
                    'idRenvois_art': row.get('idRenvois_art', 'non_classifié'),
                    'block_text': block,  # Alias pour compatibilité
                    
                    # Métadonnées supplémentaires
                    'block_index': block_idx,
                    'article_title': row.get('vedette', 'Titre inconnu'),
                    'article_head': row.get('vedette', 'Titre inconnu'),
                    'domaine': row.get('domainesEnccre', 'Non classifié'),
                    'auteur': row.get('auteurs', 'Inconnu'),
                })
    
    blocks_df = pd.DataFrame(doc_blocks)
    print(f"Nombre de blocs générés : {len(blocks_df)}")
    return blocks_df


def create_search_engine_training_pairs(df: pd.DataFrame, w2v_model, max_pairs: int = 10000) -> List[InputExample]:
    """
    Crée des paires d'entraînement Question-Passage en utilisant le modèle Word2Vec.
    Aligné avec les vrais cas d'usage : questions des utilisateurs → passages pertinents.
    
    Args:
        df: DataFrame avec les données de l'Encyclopédie 
        w2v_model: Modèle Word2Vec pré-entraîné
        max_pairs: Nombre maximum de paires à créer
        
    Returns:
        List des paires d'entraînement Question-Passage
    """
    from collections import defaultdict
    
    training_pairs = []
    
    # Templates de questions basés sur les vrais cas d'usage
    question_templates = [
        "Qu'est-ce que {concept} ?",
        "Qu'est-ce que {concept} selon l'Encyclopédie ?",
        "Comment l'Encyclopédie définit-elle {concept} ?",
        "Que dit l'Encyclopédie sur {concept} ?",
        "Comment est défini {concept} ?",
        "Quelle est la définition de {concept} ?",
        "Comment {concept} est-il traité dans l'Encyclopédie ?",
        "{concept} est-il une bonne ou une mauvaise chose ?",
        "D'où provient {concept} ?",
        "Quelles sont les causes de {concept} ?",
        "Comment fonctionne {concept} ?",
        "Quelle est l'importance de {concept} ?",
    ]
    
    print("🔍 Extraction du vocabulaire pour les questions...")
    vocabulary = set()
    article_terms = defaultdict(list)
    article_texts = {}
    
    for idx, row in df.iterrows():
        text = row.get('contenu_clean', row.get('contenu', ''))
        if pd.isna(text):
            continue
            
        # Nettoyage et tokenisation cohérente avec Word2Vec
        text_clean = re.sub(r'[^\w\s\àâäéèêëïîôöùûüÿç]', ' ', str(text).lower())
        terms = text_clean.split()
        
        # Filtrer les termes présents dans Word2Vec et pertinents pour des questions
        valid_terms = [
            term for term in terms 
            if term in w2v_model.wv.key_to_index 
            and len(term) > 3  # Mots plus longs pour des concepts
            and not term.isdigit()  # Pas de nombres
        ]
        
        vocabulary.update(valid_terms)
        article_terms[idx] = valid_terms
        article_texts[idx] = str(text)[:800]  # Texte complet pour le passage
    
    print(f"✓ Vocabulaire extrait: {len(vocabulary)} termes conceptuels")
    
    # Sélectionner les termes les plus pertinents selon Word2Vec
    print("📊 Sélection des concepts les plus riches sémantiquement...")
    concept_scores = {}
    
    for term in list(vocabulary)[:5000]:  # Limiter pour la performance
        try:
            # Score basé sur le nombre de termes similaires (richesse sémantique)
            similar_terms = w2v_model.wv.most_similar(term, topn=20)
            concept_scores[term] = len(similar_terms)
        except:
            concept_scores[term] = 0
    
    # Prendre les concepts les plus riches sémantiquement
    best_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[:1000]
    selected_concepts = [concept for concept, score in best_concepts]
    
    print(f"✓ Concepts sélectionnés: {len(selected_concepts)} termes riches")
    
    # Créer des paires Question-Passage
    print("🎯 Création de paires Question-Passage...")
    pair_count = 0
    
    for concept in selected_concepts:
        if pair_count >= max_pairs:
            break
            
        # Trouver les articles contenant ce concept
        articles_with_concept = []
        for idx, terms in article_terms.items():
            if concept in terms:
                # Score de pertinence : fréquence du terme dans l'article
                concept_frequency = terms.count(concept) / len(terms) if terms else 0
                articles_with_concept.append((idx, concept_frequency))
        
        if not articles_with_concept:
            continue
            
        # Prendre les articles les plus pertinents pour ce concept
        articles_with_concept.sort(key=lambda x: x[1], reverse=True)
        top_articles = articles_with_concept[:3]  # Top 3 articles pour ce concept
        
        # Générer des questions pour ce concept
        for template in question_templates[:4]:  # Limiter les templates
            if pair_count >= max_pairs:
                break
                
            question = template.format(concept=concept)
            
            # Créer des paires avec les meilleurs passages
            for article_idx, relevance_score in top_articles:
                if pair_count >= max_pairs:
                    break
                    
                passage = article_texts[article_idx]
                
                # Score basé sur la pertinence et la présence du concept
                normalized_score = min(0.9, 0.6 + relevance_score * 2)
                
                pair = InputExample(
                    texts=[f"Requête: {question}", f"Passage: {passage}"], 
                    label=normalized_score
                )
                training_pairs.append(pair)
                pair_count += 1
    
    # Ajouter quelques paires négatives (questions sur concepts absents)
    print("🔄 Ajout de paires négatives...")
    negative_count = min(max_pairs // 10, 1000)  # 10% de paires négatives
    
    for _ in range(negative_count):
        if len(selected_concepts) >= 2 and len(article_texts) >= 2:
            # Question sur un concept
            concept = random.choice(selected_concepts)
            question = random.choice(question_templates[:6]).format(concept=concept)
            
            # Passage ne contenant PAS ce concept
            irrelevant_articles = [
                idx for idx, terms in article_terms.items() 
                if concept not in terms
            ]
            
            if irrelevant_articles:
                article_idx = random.choice(irrelevant_articles)
                passage = article_texts[article_idx]
                
                pair = InputExample(
                    texts=[f"Requête: {question}", f"Passage: {passage}"], 
                    label=0.1  # Score très bas pour paires négatives
                )
                training_pairs.append(pair)
    
    print(f"✓ Paires Question-Passage créées: {len(training_pairs)}")
    print(f"  - Paires positives: {pair_count}")
    print(f"  - Paires négatives: {len(training_pairs) - pair_count}")
    
    return training_pairs


def create_aligned_training_pairs(blocks_df: pd.DataFrame, max_pairs: int = 10000, 
                                 questions_per_chunk: int = 2) -> List[InputExample]:
    """
    Crée des paires d'entraînement Question-Passage alignées avec le vrai use case.
    Utilise le modèle Word2Vec pour identifier les concepts et générer des questions réalistes.
    
    Args:
        blocks_df: DataFrame avec les blocs d'articles (utilisé seulement pour validation)
        max_pairs: Nombre maximum de paires à créer
        questions_per_chunk: Paramètre conservé pour compatibilité (non utilisé)
    
    Returns:
        Liste des paires d'entraînement Question-Passage
    """
    print("=== CRÉATION DES PAIRES QUESTION-PASSAGE AVEC WORD2VEC ===")
    print("🎯 Alignement avec le vrai use case : Questions utilisateurs → Passages pertinents")
    
    # Chemins vers le modèle Word2Vec créé par le notebook 1
    w2v_model_path = OUTPUT_DIR / "word2vec" / "encyclopedia_word2vec.model"
    
    if not w2v_model_path.exists():
        raise FileNotFoundError(
            f"❌ Modèle Word2Vec non trouvé: {w2v_model_path}\n"
            f"💡 Veuillez d'abord exécuter le notebook '1_creation_modele_word2vec.ipynb'"
        )
    
    # Charger les données source (même fichier que pour le modèle Word2Vec)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"❌ Fichier de données non trouvé: {DATA_FILE}")
    
    print("📝 Génération de paires Question-Passage réalistes...")
    
    try:
        # Charger les données source
        print("📁 Chargement des données source...")
        df = pd.read_csv(DATA_FILE, nrows=5000)  # Échantillon optimisé
        print(f"✓ Données chargées: {len(df)} articles")
        
        # Charger le modèle Word2Vec
        print("🧠 Chargement du modèle Word2Vec...")
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(str(w2v_model_path))
        print(f"✓ Modèle Word2Vec chargé: {len(w2v_model.wv.key_to_index)} termes")
        
        # Créer des paires Question-Passage avec le modèle Word2Vec
        question_passage_pairs = create_search_engine_training_pairs(
            df, 
            w2v_model, 
            max_pairs=max_pairs
        )
        
        if len(question_passage_pairs) == 0:
            raise ValueError("Aucune paire Question-Passage créée avec le modèle Word2Vec")
        
        print(f"✅ Paires Question-Passage créées: {len(question_passage_pairs)}")
        print("🎯 Modèle aligné pour encoder : Questions → Similarité avec Passages")
        
        return question_passage_pairs
        
    except Exception as e:
        print(f"❌ Erreur avec le modèle Word2Vec: {e}")
        raise RuntimeError(f"Impossible de créer des paires Question-Passage: {e}")


def validate_script_compatibility(blocks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valide que le DataFrame contient toutes les colonnes requises par le script.
    """
    required_columns = RAG_COMPATIBILITY["required_columns"]
    missing_columns = [col for col in required_columns if col not in blocks_df.columns]
    
    result = {
        'compatible': len(missing_columns) == 0,
        'errors': [],
        'url_format_ok': True,
        'data_types_ok': True
    }
    
    if missing_columns:
        result['errors'].append(f"Colonnes manquantes: {missing_columns}")
    
    # Vérifier le format des URLs
    if 'url' in blocks_df.columns:
        import re
        url_pattern = RAG_COMPATIBILITY["url_validation_pattern"]
        invalid_urls = blocks_df[~blocks_df['url'].str.match(url_pattern, na=False)]
        if len(invalid_urls) > 0:
            result['url_format_ok'] = False
            result['errors'].append(f"URLs invalides détectées: {len(invalid_urls)}")
    
    # Vérifier la longueur des textes
    if 'text' in blocks_df.columns:
        min_length = RAG_COMPATIBILITY["min_text_length"]
        max_length = RAG_COMPATIBILITY["max_text_length"]
        invalid_texts = blocks_df[
            (blocks_df['text'].str.len() < min_length) | 
            (blocks_df['text'].str.len() > max_length)
        ]
        if len(invalid_texts) > 0:
            result['data_types_ok'] = False
            result['errors'].append(f"Textes de longueur invalide: {len(invalid_texts)}")
    
    return result


def generate_final_outputs(blocks_df: pd.DataFrame, model, batch_size: int = 32) -> Dict[str, str]:
    """
    Génère tous les fichiers de sortie finaux pour la compatibilité RAG.
    
    Args:
        blocks_df: DataFrame avec les blocs préparés
        model: Modèle SentenceTransformer
        batch_size: Taille de batch pour la génération d'embeddings
    
    Returns:
        Dictionnaire avec les chemins des fichiers générés
    """
    output_files = {}
    
    # 1. Sauvegarder les blocs avec colonnes standardisées
    blocks_file = OUTPUT_FILES["blocks"]
    blocks_df.to_csv(blocks_file, index=False)
    output_files["blocks"] = str(blocks_file)
    
    # 2. Générer les embeddings
    print("Génération des embeddings...")
    embeddings = []
    total_batches = (len(blocks_df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(blocks_df), batch_size), 
                  desc="Génération embeddings", total=total_batches):
        batch_texts = blocks_df.iloc[i:i+batch_size]['text'].tolist()
        batch_embeddings = model.encode(
            batch_texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=batch_size
        )
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # Sauvegarder les embeddings
    embeddings_file = OUTPUT_FILES["embeddings"]
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    output_files["embeddings"] = str(embeddings_file)
    
    # 3. Créer l'index de recherche
    print("Création de l'index de recherche...")
    from sklearn.neighbors import NearestNeighbors
    
    search_index = NearestNeighbors(
        n_neighbors=SEARCH_PARAMS["top_k"],
        algorithm='auto',
        metric='cosine'
    )
    search_index.fit(embeddings)
    
    # Sauvegarder l'index
    index_file = OUTPUT_FILES["index"]
    with open(index_file, 'wb') as f:
        pickle.dump(search_index, f)
    output_files["index"] = str(index_file)
    
    return output_files


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données brutes et prépare les blocs pour l'entraînement.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (données brutes, blocs préparés)
    """
    # Charger les données brutes
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Le fichier {DATA_FILE} n'existe pas.")
    
    df_raw = pd.read_csv(DATA_FILE)
    print(f"Données brutes chargées: {len(df_raw)} articles")
    
    # Préparer les blocs avec colonnes standardisées
    blocks_df = prepare_article_blocks(df_raw)
    print(f"Blocs préparés: {len(blocks_df)} blocs")
    
    return df_raw, blocks_df
