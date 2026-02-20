"""
title: Encycloscope
author: Morgan Blangeois
version: 3.0
requirements: openai>=1.44.1, sentence-transformers>=2.2.0, llama-index, llama-index-llms-openai, pandas, scikit-learn
"""

# =========================================
# Ajout des imports
# =========================================
import os
import pickle
import numpy as np
import pandas as pd
import traceback
from typing import List, Dict, Optional, Sequence
from collections import defaultdict
from pydantic import BaseModel, Field

# SentenceTransformers et Scikit-learn pour la recherche initiale
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Client OpenAI Async
from openai import AsyncOpenAI

# Composants LlamaIndex
from llama_index.core import Response, Settings, QueryBundle
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    ContextRelevancyEvaluator,
    EvaluationResult,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI as LlamaOpenAI


# =========================================
# Fonction pour extraire l'historique
# =========================================
def get_conversation_history(messages: List[Dict]) -> List[Dict]:
    """
    Transforme la liste 'messages' en une structure compatible avec OpenAI
    pour conserver le contexte de conversation multi-turn.
    """
    conversation_history = []
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            conversation_history.append(
                {"role": msg["role"], "content": msg["content"]}
            )
    return conversation_history


# =========================================
# Fonction pour formatter les sources proprement
# =========================================
def _unique_formatted_sources(nodes: Sequence[NodeWithScore]) -> List[str]:
    seen = set()
    out = []

    for nws in nodes:
        meta = nws.node.metadata
        url = meta.get("url") or ""
        key = url or meta.get("article_id") or meta.get("title")
        if key in seen:
            continue
        seen.add(key)

        title = meta.get("title", "Inconnu")
        author = str(meta.get("author", "Inconnu") or "Inconnu")
        if author.lower() == "nan":
            author = "Inconnu"

        out.append(f"[{title} ({author})]({url})")

    return out


# =========================================
# Classe Pipe principale
# =========================================
class Pipe:
    """
    Pipe RAG sur l'Encyclopédie avec post-processing et évaluations.
    """

    class Valves(BaseModel):
        USE_OPEN_WEBUI_INTERNAL_MODELS: bool = Field(
            default=False,
            description="Activer pour utiliser les modèles internes d'Open-WebUI.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="Clé API OpenAI. Non utilisée si le mode interne est activé.",
        )
        OPEN_WEBUI_API_KEY: str = Field(
            default="ollama",
            description="Clé API à utiliser pour les modèles internes à trouver dans l'interface d'Open-WebUI.",
        )
        OPEN_WEBUI_BASE_URL: str = Field(
            default="http://localhost:8080/api",
            description="URL de base pour l'API Open-WebUI.",
        )
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="URL de base pour l'API OpenAI.",
        )
        LOCAL_MODEL_ID: str = Field(
            default="llama3.2:3b",
            description="ID du modèle local à utiliser avec Open-WebUI. Ex: 'gemma3n:e4b'.",
        )
        MODEL_ID: str = Field(
            default="gpt-4o",
            description="ID du modèle OpenAI à utiliser. Ex: 'gpt-4o'.",
        )
        EVAL_MODEL_ID: str = Field(
            default="gpt-4o-mini",
            description="ID du modèle pour les évaluations. Doit être compatible avec l'API configurée (OpenAI ou interne).",
        )
        MODEL_PATH: str = Field(
            default="data/checkpoint-5048",
            description="Chemin modèle embeddings SentenceTransformer",
        )
        INDEX_PATH: str = Field(
            default="data/encyclopedia_index.pkl",
            description="Chemin index NearestNeighbors (.pkl)",
        )
        BLOCKS_PATH: str = Field(
            default="data/encyclopedia_blocks.csv",
            description="Chemin métadonnées blocs (.csv)",
        )
        TOP_K_RESULTS: int = Field(
            default=10, description="Nombre de blocs à récupérer initialement"
        )
        SIMILARITY_THRESHOLD: float = Field(
            default=0.55,
            description="Seuil de similarité (cosinus) pour filtrer les nœuds récupérés (0.0 à 1.0)",
        )
        MAX_TOKENS: int = Field(
            default=4096, description="Max tokens pour la réponse générée"
        )
        TEMPERATURE: float = Field(default=0.1, description="Température de génération")
        MAX_BLOCKS_PER_ARTICLE: int = Field(
            default=3,
            description="Nombre maximal de blocs autorisés par article avant de passer au suivant",
        )
        MULTITURN_CONTEXT: bool = Field(
            default=True, description="Gérer l'historique conversationnel"
        )
        EVALUATE_FAITHFULNESS: bool = Field(
            default=True,
            description="Activer l'évaluation de la fidélité (réponse vs contexte)",
        )
        EVALUATE_CONTEXT_RELEVANCY: bool = Field(
            default=True,
            description="Activer l'évaluation de la pertinence (contexte vs question)",
        )

    def __init__(self):
        """Initialise les attributs de la classe Pipe."""
        self.type = "pipe"
        self.name = "Encycloscope 3.0"
        self.valves = self.Valves()
        self.version = "3.0"

        # Clients et modèles initialisés à None
        self.openai_client: Optional[AsyncOpenAI] = None
        self.model: Optional[SentenceTransformer] = None
        self.index = None
        self.blocks_df: Optional[pd.DataFrame] = None

        # Composants LlamaIndex initialisés à None
        self.eval_llm: Optional[LlamaOpenAI] = None
        self.faithfulness_evaluator: Optional[FaithfulnessEvaluator] = None
        self.context_relevancy_evaluator: Optional[ContextRelevancyEvaluator] = None
        self.last_config_signature: Optional[tuple] = None
        print(f"[{self.name}] Instance créée (version {self.version})")

    async def initialize(self, __event_emitter__=None):
        """Charge les modèles, index, données et initialise clients et évaluateurs."""
        print(f"[{self.name}] Initialisation...")
        api_key_to_use = self.valves.OPENAI_API_KEY
        base_url_to_use = self.valves.OPENAI_BASE_URL

        if self.valves.USE_OPEN_WEBUI_INTERNAL_MODELS:
            print(f"[{self.name}] Mode 'Modèles Internes Open-WebUI' activé.")
            base_url_to_use = self.valves.OPEN_WEBUI_BASE_URL
            api_key_to_use = self.valves.OPEN_WEBUI_API_KEY
            print(f"[{self.name}] > URL de base configurée à: {base_url_to_use}")
            print(f"[{self.name}] > Clé API configurée pour l'interne.")
            print(f"[{self.name}] > Modèle de génération: {self.valves.MODEL_ID}")
            print(f"[{self.name}] > Modèle d'évaluation: {self.valves.EVAL_MODEL_ID}")
            print(
                f"[{self.name}] > Modèle local configuré: {self.valves.LOCAL_MODEL_ID}"
            )
        else:
            print(f"[{self.name}] Mode OpenAI standard activé.")
            print(f"[{self.name}] > URL de base: {base_url_to_use}")
            print(f"[{self.name}] > Modèle de génération: {self.valves.MODEL_ID}")
            print(f"[{self.name}] > Modèle d'évaluation: {self.valves.EVAL_MODEL_ID}")

        print(f"[{self.name}] Répertoire actuel : {os.getcwd()}")
        print(
            f"[{self.name}] Chemin modèle embeddings : {os.path.abspath(self.valves.MODEL_PATH)}"
        )
        print(f"[{self.name}] Chemin index : {os.path.abspath(self.valves.INDEX_PATH)}")
        print(
            f"[{self.name}] Chemin blocs : {os.path.abspath(self.valves.BLOCKS_PATH)}"
        )

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Vérification configuration API...",
                        "done": False,
                    },
                }
            )
            if not api_key_to_use:
                raise ValueError("La clé API est manquante.")

            # 1. Initialisation Client OpenAI Async
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Init Client OpenAI...", "done": False},
                }
            )
            self.openai_client = AsyncOpenAI(
                api_key=api_key_to_use,
                base_url=base_url_to_use,
            )
            print(
                f"[{self.name}] Client AsyncOpenAI (génération) initialisé pour l'URL: {base_url_to_use}"
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Client OpenAI prêt.", "done": True},
                }
            )

            # 2. Initialisation LlamaIndex (LLM et Évaluateurs)
            needs_llama_eval = (
                self.valves.EVALUATE_FAITHFULNESS
                or self.valves.EVALUATE_CONTEXT_RELEVANCY
            )
            if needs_llama_eval:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Initialisation LlamaIndex...",
                            "done": False,
                        },
                    }
                )
                # Sélection du modèle d'évaluation selon le mode
                eval_model_to_use = self.valves.EVAL_MODEL_ID
                self.eval_llm = LlamaOpenAI(
                    model=eval_model_to_use,
                    api_key=api_key_to_use,
                    api_base=base_url_to_use,
                )
                Settings.llm = self.eval_llm
                print(
                    f"[{self.name}] LLM LlamaIndex ({eval_model_to_use}) initialisé pour évaluations."
                )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "LLM Évaluation prêt.", "done": True},
                    }
                )

                if self.valves.EVALUATE_FAITHFULNESS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Initialisation Évaluateur Faithfulness...",
                                "done": False,
                            },
                        }
                    )
                    self.faithfulness_evaluator = FaithfulnessEvaluator(
                        llm=self.eval_llm
                    )
                    print(f"[{self.name}] Évaluateur Faithfulness initialisé.")
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Evaluateur Fidélité prêt.",
                                "done": True,
                            },
                        }
                    )

                if self.valves.EVALUATE_CONTEXT_RELEVANCY:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Initialisation Évaluateur ContextRelevancy...",
                                "done": False,
                            },
                        }
                    )
                    self.context_relevancy_evaluator = ContextRelevancyEvaluator(
                        llm=self.eval_llm
                    )
                    print(f"[{self.name}] Évaluateur ContextRelevancy initialisé.")
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Evaluateur ContextRelevancy prêt.",
                                "done": True,
                            },
                        }
                    )

            # 3. Chargement Modèle Embeddings
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Chargement Embeddings...", "done": False},
                }
            )
            if not os.path.exists(self.valves.MODEL_PATH):
                raise FileNotFoundError(
                    f"Modèle Embeddings non trouvé: {self.valves.MODEL_PATH}"
                )
            self.model = SentenceTransformer(self.valves.MODEL_PATH)
            print(f"[{self.name}] Modèle Embeddings chargé: {self.valves.MODEL_PATH}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Embeddings chargés.", "done": True},
                }
            )

            # 4. Chargement Index NearestNeighbors
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Chargement Index...", "done": False},
                }
            )
            if not os.path.exists(self.valves.INDEX_PATH):
                raise FileNotFoundError(f"Index non trouvé: {self.valves.INDEX_PATH}")
            with open(self.valves.INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
            print(f"[{self.name}] Index chargé: {self.valves.INDEX_PATH}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Index chargé.", "done": True},
                }
            )

            # 5. Chargement Métadonnées Blocs
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Chargement Métadonnées...", "done": False},
                }
            )
            if not os.path.exists(self.valves.BLOCKS_PATH):
                raise FileNotFoundError(
                    f"Métadonnées non trouvées: {self.valves.BLOCKS_PATH}"
                )
            self.blocks_df = pd.read_csv(self.valves.BLOCKS_PATH)
            self.blocks_df["auteur"] = (
                self.blocks_df["auteur"].fillna("Inconnu").astype(str).str.strip()
            )
            required_cols = ["article_head", "block_text", "articleID"]
            if not all(col in self.blocks_df.columns for col in required_cols):
                print(
                    f"[{self.name}] ATTENTION: Colonnes manquantes dans {self.valves.BLOCKS_PATH}. Requis: {required_cols}. Présentes: {self.blocks_df.columns.tolist()}"
                )
            print(
                f"[{self.name}] Métadonnées chargées: {self.valves.BLOCKS_PATH} ({len(self.blocks_df)} blocs)"
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Métadonnées chargées.", "done": True},
                }
            )

            print(f"[{self.name}] Initialisation terminée avec succès.")
            self.last_config_signature = self._get_config_signature()
        except FileNotFoundError as fnf_e:
            print(f"[{self.name}] ERREUR Fichier Non Trouvé: {fnf_e}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"ERREUR Fichier: {fnf_e}. Vérifiez les chemins.",
                        "done": True,
                        "error": True,
                    },
                }
            )
            raise
        except Exception as e:
            print(f"[{self.name}] ERREUR Init Générale: {e}\n{traceback.format_exc()}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"ERREUR Init: {e}",
                        "done": True,
                        "error": True,
                    },
                }
            )
            raise

    def _get_config_signature(self) -> tuple:
        """Génère une signature unique pour la configuration API actuelle."""
        if self.valves.USE_OPEN_WEBUI_INTERNAL_MODELS:
            # En mode interne, la signature dépend du mode, de l'URL locale et des modèles
            base_url = self.valves.OPEN_WEBUI_BASE_URL
            api_key = self.valves.OPEN_WEBUI_API_KEY
            return (
                True,  # Indicateur du mode
                base_url,
                api_key,
                self.valves.MODEL_ID,
                self.valves.EVAL_MODEL_ID,
                self.valves.LOCAL_MODEL_ID,
            )
        else:
            # En mode OpenAI, la signature dépend des valves OpenAI
            return (
                False,  # Indicateur du mode
                self.valves.OPENAI_BASE_URL,
                self.valves.OPENAI_API_KEY,
                self.valves.MODEL_ID,
                self.valves.EVAL_MODEL_ID,
            )

    def _truncate_context(self, text: str, max_tokens: int = 125000) -> str:
        """Tronque le texte (méthode basique basée sur les espaces)."""
        tokens = text.split()
        if len(tokens) > max_tokens:
            print(f"[{self.name}] Contexte tronqué à {max_tokens} tokens (approx).")
            return " ".join(tokens[:max_tokens])
        return text

    async def search_encyclopedia(
        self, query: str, k: int, __event_emitter__=None
    ) -> List[Dict]:
        """Encode la requête et recherche les K blocs similaires."""
        print(f"[{self.name}] Recherche initiale (k={k}) pour: '{query[:50]}...'")
        if not self.model or self.index is None or self.blocks_df is None:
            raise RuntimeError("Recherche impossible: composants non initialisés.")

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Encodage requête...", "done": False},
                }
            )
            query_vector = self.model.encode([query], convert_to_numpy=True)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Recherche des {k} plus proches...",
                        "done": False,
                    },
                }
            )
            distances, indices = self.index.kneighbors(query_vector, n_neighbors=k)
            scores = 1.0 - distances[0]
            results_dict_list = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.blocks_df):
                    block = self.blocks_df.iloc[idx].to_dict()
                    result_item = {
                        "score": float(scores[i]),
                        "title": block.get("article_head", "Titre inconnu"),
                        "domain": block.get("domaine", "Non classifié"),
                        "author": block.get("auteur", "Inconnu"),
                        "articleID": block.get("articleID", ""),
                        "text": block.get("block_text", ""),
                        "url": f"http://enccre.academie-sciences.fr/encyclopedie/article/{block.get('articleID', '')}/",
                    }
                    result_item.update(
                        {k: v for k, v in block.items() if k not in result_item}
                    )
                    results_dict_list.append(result_item)
                else:
                    print(f"[{self.name}] Attention: Index invalide {idx} ignoré.")
            print(
                f"[{self.name}] Recherche initiale: {len(results_dict_list)} résultats trouvés."
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Recherche initiale terminée.",
                        "done": True,
                    },
                }
            )
            return results_dict_list
        except Exception as e:
            print(f"[{self.name}] ERREUR Recherche: {e}\n{traceback.format_exc()}")
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"**Erreur interne Recherche:** {e}",
                        "done": True,
                    },
                }
            )
            return []

    async def generate_response(
        self,
        conversation_history: List[Dict],
        relevant_nodes_with_score: Sequence[NodeWithScore],
        __event_emitter__=None,
    ) -> str:
        """Génère une réponse LLM basée sur les nœuds pertinents fournis."""
        num_nodes = len(relevant_nodes_with_score)
        print(f"[{self.name}] Génération réponse depuis {num_nodes} nœud(s) filtré(s).")
        if not self.openai_client:
            raise RuntimeError("Client OpenAI non initialisé.")

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Préparation contexte LLM...",
                        "done": False,
                    },
                }
            )
            context_docs_str = "\n\n---\n\n".join(
                f"**Extrait de: {node_with_score.node.metadata.get('title', 'Inconnu')} "
                f"(Auteur: {node_with_score.node.metadata.get('author', 'Inconnu')})**\n"
                f"\n{node_with_score.node.text}"
                for node_with_score in relevant_nodes_with_score
            )
            system_prompt = (
                f"Extraits de l'Encyclopédie de Diderot et d'Alembert:\n{context_docs_str}\n\n"
                "NE VOUS APPUYEZ QUE SUR CES EXTRAITS POUR RÉPONDRE À L'UTILISATEUR.\n\n"
                "En vous basant sur le contexte extrait de l'Encyclopédie de Diderot et d'Alembert,"
                "veuillez répondre à la requête. Votre réponse doit :\n"
                "1. Synthétiser les informations pertinentes des différents articles.\n"
                "2. Mettre en évidence les liens entre les concepts et les idées des différents auteurs.\n"
                "3. Contextualiser la réponse dans le cadre des idées des Lumières du 18ème siècle, tout en s'affranchissant du romantisme autours de cette période et de ses auteurs.\n"
                "4. Ne s'appuyer que sur le contexte fourni.\n"
                "5. Préciser si le contexte ne fourni pas d'information précise répondant à la question de l'utilisateur.\n"
                "6. Citez les documents pertinents en utilisant le format suivant : [Titre (Auteur)](URL) en pleine phrase où cela est nécessaire.\n"
                "7. Citez des passages directement issus des extraits fournis de façon naturelle dans votre réponse."
                "8. Répondez dans la même langue que l'utilisateur."
                "FORMAT DE RÉPONSE REQUIS:\n"
                "### Synthèse des articles\n"
                "...\n"
                "### Analyse des concepts\n"
                "...\n"
            )
            messages_for_api = [
                {"role": "system", "content": system_prompt}
            ] + conversation_history

            # Sélection du modèle de génération selon le mode
            generation_model_to_use = (
                self.valves.LOCAL_MODEL_ID
                if self.valves.USE_OPEN_WEBUI_INTERNAL_MODELS
                else self.valves.MODEL_ID
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Appel LLM ({generation_model_to_use})...",
                        "done": False,
                    },
                }
            )
            response = await self.openai_client.chat.completions.create(
                model=generation_model_to_use,
                messages=messages_for_api,
                max_tokens=self.valves.MAX_TOKENS,
                temperature=self.valves.TEMPERATURE,
                stream=False,
            )
            final_answer = response.choices[0].message.content.strip()
            print(f"[{self.name}] Réponse LLM générée.")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Réponse LLM reçue.", "done": True},
                }
            )
            return final_answer
        except Exception as e:
            print(f"[{self.name}] ERREUR Génération: {e}\n{traceback.format_exc()}")
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"**Erreur interne Génération:** {e}",
                        "done": True,
                    },
                }
            )
            return f"Erreur lors de la génération de la réponse ({type(e).__name__})."

    async def pipe(self, body: dict, user_data: dict = None, __event_emitter__=None):
        """Orchestre recherche, post-processing, évaluations, et génération."""
        print(f"[{self.name}] Exécution Pipe...")
        start_time = pd.Timestamp.now()
        user_data = user_data or {}

        current_signature = self._get_config_signature()
        if self.last_config_signature != current_signature:
            print(
                f"[{self.name}] La configuration a changé ou n'est pas initialisée. Ré-initialisation..."
            )
            try:
                await self.initialize(__event_emitter__)
            except Exception as init_e:
                error_msg = f"Erreur lors de la ré-initialisation: {init_e}"
                print(f"[{self.name}] {error_msg}")
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": f"**{error_msg}**", "done": True},
                    }
                )
                return f"**Erreur Pipe:** {error_msg}"

        if not self.openai_client or not self.model or not self.index:
            error_msg = "Les composants de la pipe ne sont pas initialisés correctement. Veuillez vérifier les logs."
            print(f"[{self.name}] {error_msg}")
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"**{error_msg}**", "done": True},
                }
            )
            return f"**Erreur Pipe:** {error_msg}"

        messages = body.get("messages", [])
        if not messages:
            return "**Erreur Pipe:** Pas de messages."

        conversation_history = (
            get_conversation_history(messages)
            if self.valves.MULTITURN_CONTEXT
            else [messages[-1]]
        )
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        if not user_query:
            return "**Erreur Pipe:** Requête utilisateur non trouvée."
        print(f"[{self.name}] Requête: '{user_query[:100]}...'")

        try:
            initial_results_dicts = await self.search_encyclopedia(
                user_query, self.valves.TOP_K_RESULTS, __event_emitter__
            )
            if self.valves.MAX_BLOCKS_PER_ARTICLE > 0:
                limited_results = []
                seen_per_article = defaultdict(int)
                for item in initial_results_dicts:
                    art_id = (
                        item.get("article_id")
                        or item.get("articleID")
                        or item.get("url")
                        or item.get("title")
                    )
                    if seen_per_article[art_id] < self.valves.MAX_BLOCKS_PER_ARTICLE:
                        limited_results.append(item)
                        seen_per_article[art_id] += 1
                initial_results_dicts = limited_results
                print(
                    f"[{self.name}] Post-filtre article: {len(initial_results_dicts)} blocs retenus (max {self.valves.MAX_BLOCKS_PER_ARTICLE}/article)"
                )
            if not initial_results_dicts:
                return "Aucun passage trouvé dans l'Encyclopédie pour cette question."

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Conversion et Filtrage...", "done": False},
                }
            )
            nodes_with_score_from_search = []
            for node_data in initial_results_dicts:
                if node_data.get("text"):
                    metadata = {
                        k: v for k, v in node_data.items() if k not in ["text", "score"]
                    }
                    text_node = TextNode(text=node_data["text"], metadata=metadata)
                    nodes_with_score_from_search.append(
                        NodeWithScore(node=text_node, score=node_data.get("score", 0.0))
                    )

            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=self.valves.SIMILARITY_THRESHOLD
            )
            query_bundle = QueryBundle(query_str=user_query)
            filtered_nodes_with_score = postprocessor.postprocess_nodes(
                nodes=nodes_with_score_from_search, query_bundle=query_bundle
            )
            num_after_filter = len(filtered_nodes_with_score)
            print(
                f"[{self.name}] Post-Proc: {num_after_filter} nœuds (sur {len(nodes_with_score_from_search)}) retenus (seuil: {self.valves.SIMILARITY_THRESHOLD})."
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"{num_after_filter} extrait(s) pertinent(s) trouvé(s).",
                        "done": True,
                    },
                }
            )
            if not filtered_nodes_with_score:
                return "Aucun passage jugé suffisamment similaire trouvé dans l'Encyclopédie après filtrage."

            final_nodes_for_llm = filtered_nodes_with_score
            context_relevancy_str = ""
            if (
                self.valves.EVALUATE_CONTEXT_RELEVANCY
                and self.context_relevancy_evaluator
            ):
                print(f"[{self.name}] Éval Pertinence Contexte...")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Éval Pertinence Ctx...",
                            "done": False,
                        },
                    }
                )
                try:
                    contexts_texts = [n.node.text for n in final_nodes_for_llm]
                    relevancy_eval: EvaluationResult = (
                        await self.context_relevancy_evaluator.aevaluate(
                            query=user_query, contexts=contexts_texts
                        )
                    )
                    relevancy_score = relevancy_eval.score
                    is_relevant = relevancy_eval.passing
                    print(
                        f"[{self.name}] Résultat Pertinence Ctx: Score={relevancy_score}, Passing={is_relevant}"
                    )
                    context_relevancy_str = "**Adéquation du contexte :** "
                    if relevancy_score is not None:
                        if relevancy_score > 0.75:
                            context_relevancy_str += f"✅ Les extraits trouvés semblent bien correspondre à votre question (Score: {relevancy_score:.2f})."
                        elif relevancy_score > 0.5:
                            context_relevancy_str += f"🟠 Les extraits trouvés sont moyennement liés à votre question (Score: {relevancy_score:.2f})."
                        else:
                            context_relevancy_str += f"⚠️ Les extraits trouvés correspondent peu à votre question (Score: {relevancy_score:.2f}). Reformuler pourrait aider."
                    elif is_relevant is not None:
                        context_relevancy_str += f"{'✅ Les extraits semblent pertinents.' if is_relevant else '⚠️ La pertinence des extraits est douteuse.'}"
                    else:
                        context_relevancy_str += (
                            "Évaluation de l'adéquation non concluante."
                        )
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Éval Pertinence Ctx terminée.",
                                "done": True,
                            },
                        }
                    )
                except Exception as ctx_eval_e:
                    print(
                        f"[{self.name}] ERREUR Eval Pertinence Ctx: {ctx_eval_e}\n{traceback.format_exc()}"
                    )
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"**Erreur Éval Pertinence Ctx:** {ctx_eval_e}",
                                "done": True,
                            },
                        }
                    )
                    context_relevancy_str = (
                        "**Pertinence Contexte :** Évaluation échouée."
                    )

            final_answer = await self.generate_response(
                conversation_history, final_nodes_for_llm, __event_emitter__
            )
            faithfulness_score_str = ""
            if self.valves.EVALUATE_FAITHFULNESS and self.faithfulness_evaluator:
                print(f"[{self.name}] Éval Fidélité Réponse...")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Éval Fidélité Réponse...",
                            "done": False,
                        },
                    }
                )
                try:
                    response_obj = Response(
                        response=final_answer, source_nodes=final_nodes_for_llm
                    )
                    print(
                        f"[{self.name}] Appel aevaluate_response pour fidélité (nodes: {len(response_obj.source_nodes)})"
                    )
                    faith_eval: EvaluationResult = (
                        await self.faithfulness_evaluator.aevaluate_response(
                            response=response_obj
                        )
                    )
                    is_faithful = faith_eval.passing
                    feedback = faith_eval.feedback
                    print(
                        f"[{self.name}] Résultat Fidélité: Passing={is_faithful}, Feedback='{feedback}'"
                    )
                    faithfulness_score_str = "**Fiabilité de la réponse :** "
                    if is_faithful:
                        faithfulness_score_str += "✅ La réponse semble bien basée uniquement sur les extraits de l'Encyclopédie fournis."
                    else:
                        faithfulness_score_str += "⚠️ La réponse contient potentiellement des informations ne provenant pas des extraits de l'Encyclopédie fournis."
                        if feedback:
                            cleaned_feedback = (
                                feedback.lstrip("NO:").lstrip("YES:").strip()
                            )
                            if cleaned_feedback:
                                faithfulness_score_str += (
                                    f"\n_Raison possible : {cleaned_feedback}_"
                                )
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Éval Fidélité terminée.",
                                "done": True,
                            },
                        }
                    )
                except Exception as faith_eval_e:
                    print(
                        f"[{self.name}] ERREUR Eval Fidélité: {faith_eval_e}\n{traceback.format_exc()}"
                    )
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"**Erreur Éval Fidélité:** {faith_eval_e}",
                                "done": True,
                            },
                        }
                    )
                    faithfulness_score_str = (
                        "**Fiabilité Réponse :** Évaluation échouée."
                    )

            output_parts = [final_answer]

            # Debug des évaluations
            print(
                f"[{self.name}] DEBUG - context_relevancy_str: '{context_relevancy_str}'"
            )
            print(
                f"[{self.name}] DEBUG - faithfulness_score_str: '{faithfulness_score_str}'"
            )

            if final_nodes_for_llm:
                formatted_sources_list = _unique_formatted_sources(final_nodes_for_llm)
                sources_section_str = "### Sources\n" + ", ".join(
                    formatted_sources_list
                )
                output_parts.append(sources_section_str)
            if context_relevancy_str or faithfulness_score_str:
                print(f"[{self.name}] DEBUG - Ajout des évaluations au message final")
                output_parts.append("---")
                if context_relevancy_str:
                    output_parts.append(context_relevancy_str)
                if faithfulness_score_str:
                    output_parts.append(faithfulness_score_str)
            else:
                print(
                    f"[{self.name}] DEBUG - Aucune évaluation à ajouter (chaînes vides)"
                )
            signature_str = "_Moteur de recherche conceptuel basé sur l'Encyclopédie de Diderot et d'Alembert_"
            output_parts.append(signature_str)
            output_message = "\n\n".join(output_parts)
            print(
                f"[{self.name}] Exécution Pipe terminée en {pd.Timestamp.now() - start_time}."
            )
            return output_message.strip()

        except Exception as e:
            print(f"[{self.name}] ERREUR Majeure Pipe: {e}\n{traceback.format_exc()}")
            error_msg = (
                f"**Erreur Critique Pipe Encycloscope:**\n{type(e).__name__}: {e}"
            )
            await __event_emitter__(
                {"type": "message", "data": {"content": error_msg, "done": True}}
            )
            return error_msg
