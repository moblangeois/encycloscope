"""
DAlemBERT Encoder pour l'Encyclopédie de Diderot et d'Alembert
Wrapper pour utiliser d'AlemBERT avec une API compatible SentenceTransformers

d'AlemBERT est un modèle RoBERTa pré-entraîné sur le français du 17e-18e siècle,
optimisé pour les textes historiques.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc


class DAlemBERTEncoder:
    """
    Wrapper pour utiliser d'AlemBERT avec sentence embeddings.
    d'AlemBERT est un modèle RoBERTa, pas sentence-transformers,
    donc on crée un encodeur custom avec mean pooling.
    """

    def __init__(self, model_name: str = 'pjox/dalembert', pooling: str = 'mean'):
        """
        Args:
            model_name: 'pjox/dalembert' pour le modèle historique français
            pooling: 'mean', 'cls', ou 'max' pour agréger les tokens
        """
        print(f"[DAlemBERTEncoder] Chargement de {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.model_name = model_name

        # Détecter le device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Désactiver les gradients pour économiser la mémoire
        torch.set_num_threads(1)
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[DAlemBERTEncoder] Modèle chargé sur {self.device}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling sur les token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _cls_pooling(self, model_output):
        """Utilise le token [CLS]"""
        return model_output[0][:, 0, :]

    def _max_pooling(self, model_output, attention_mask):
        """Max pooling sur les token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1)[0]

    def encode(
        self,
        texts,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode des textes en embeddings.
        API compatible avec SentenceTransformers.

        Args:
            texts: Un texte ou une liste de textes
            batch_size: Taille des batches (16 recommandé pour d'AlemBERT)
            show_progress_bar: Afficher la progression
            convert_to_numpy: Convertir en numpy array
            normalize_embeddings: Normaliser les vecteurs (L2)

        Returns:
            Embeddings de shape (n_texts, hidden_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress_bar and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="[DAlemBERT] Encoding")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]

                try:
                    # Tokenize
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}

                    # Forward pass
                    model_output = self.model(**encoded)

                    # Pooling
                    if self.pooling == 'mean':
                        embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
                    elif self.pooling == 'cls':
                        embeddings = self._cls_pooling(model_output)
                    else:  # max
                        embeddings = self._max_pooling(model_output, encoded['attention_mask'])

                    # Normaliser si demandé
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    all_embeddings.append(embeddings.cpu())

                    # Nettoyer la mémoire
                    del encoded, model_output, embeddings
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # Garbage collection périodique
                    if i % (batch_size * 50) == 0:
                        gc.collect()

                except Exception as e:
                    print(f"\n[DAlemBERT] Erreur au batch {i//batch_size}: {e}")
                    # En cas d'erreur, réessayer avec batch_size=1
                    if batch_size > 1:
                        print("[DAlemBERT] Retry avec batch_size=1...")
                        return self.encode(
                            texts,
                            batch_size=1,
                            show_progress_bar=show_progress_bar,
                            convert_to_numpy=convert_to_numpy,
                            normalize_embeddings=normalize_embeddings
                        )
                    raise

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings (768 pour d'AlemBERT)"""
        return self.model.config.hidden_size
