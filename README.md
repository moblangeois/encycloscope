# Encycloscope

Moteur de recherche conversationnel pour l'Encyclopédie de Diderot et d'Alembert, basé sur une architecture RAG (Retrieval-Augmented Generation).

L'Encycloscope permet d'interroger en langage naturel le corpus intégral de l'Encyclopédie (édition ENCCRE) et de recevoir des réponses synthétiques sourcées, accompagnées d'évaluations automatiques de fidélité et de pertinence.

## Architecture

Le pipeline repose sur trois composants principaux :

1. **Recherche sémantique** : un modèle SentenceTransformers fine-tuné sur le corpus encyclopédique encode les requêtes et retrouve les passages pertinents via un index NearestNeighbors.
2. **Post-traitement** : les résultats sont filtrés par seuil de similarité cosinus et par limite de blocs par article (LlamaIndex `SimilarityPostprocessor`).
3. **Génération** : un LLM (OpenAI GPT-4o par défaut, compatible avec des modèles locaux via Open-WebUI) produit une réponse synthétique contrainte aux seuls extraits retrouvés.

Deux évaluateurs LlamaIndex (`FaithfulnessEvaluator`, `ContextRelevancyEvaluator`) vérifient automatiquement que la réponse est fidèle au contexte et que les extraits sont pertinents par rapport à la question.

## Données

Le corpus provient de l'édition critique en ligne de l'Encyclopédie ([ENCCRE](http://enccre.academie-sciences.fr/)). Les données pré-traitées (blocs de texte, index d'embeddings, métadonnées) ne sont pas incluses dans ce dépôt. Contactez l'auteur pour y accéder.

Fichiers attendus dans `data/` :

| Fichier | Description |
|---|---|
| `checkpoint-5048/` | Modèle SentenceTransformers fine-tuné |
| `encyclopedia_index.pkl` | Index NearestNeighbors (pickle) |
| `encyclopedia_blocks.csv` | Métadonnées et texte des blocs |

## Installation

```bash
pip install openai sentence-transformers llama-index llama-index-llms-openai pandas scikit-learn
```

## Configuration

Le système fonctionne comme un pipe [Open-WebUI](https://github.com/open-webui/open-webui). Les paramètres sont configurables via les Valves :

| Paramètre | Défaut | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Clé API OpenAI |
| `MODEL_ID` | `gpt-4o` | Modèle de génération |
| `EVAL_MODEL_ID` | `gpt-4o-mini` | Modèle d'évaluation |
| `TOP_K_RESULTS` | 10 | Nombre de blocs retrouvés |
| `SIMILARITY_THRESHOLD` | 0.55 | Seuil de similarité cosinus |
| `MAX_BLOCKS_PER_ARTICLE` | 3 | Limite de blocs par article |
| `USE_OPEN_WEBUI_INTERNAL_MODELS` | False | Mode modèles locaux |

## Utilisation

L'Encycloscope est conçu pour fonctionner comme un pipe dans Open-WebUI. Il peut aussi être intégré dans d'autres applications via la classe `Pipe`.

```python
pipe = Pipe()
await pipe.initialize(__event_emitter__=emitter)

body = {"messages": [{"role": "user", "content": "Que dit l'Encyclopédie sur le commerce ?"}]}
response = await pipe.pipe(body, __event_emitter__=emitter)
```

## Contexte académique

L'Encycloscope est un artefact de recherche développé dans le cadre d'une collaboration entre Clermont Recherche Management (CleRMa, Université Clermont Auvergne) et le laboratoire Philosophies et Rationalités (PHIER).

### Publications associées

- Blangeois, M., Sabatier, N., Vasile, A. & Galinon, H. (2025). Explorer l'émergence de la science économique dans l'Encyclopédie grâce aux LLM : retours d'expérience sur l'Encycloscope 2.0. Communication présentée au colloque DH@LLM, Institut d'Études Avancées de Paris & Sorbonne Université.
- Blangeois, M., Vasile, A., Sabatier, N. & Galinon, H. (2024). L'exploration de l'Encyclopédie par l'intelligence artificielle : problèmes, méthodes, premiers résultats. Journée « Qu'est-ce que l'IA peut faire pour vous ? », Maison des Sciences de l'Homme, Clermont-Ferrand. [hal-04928888](https://hal.science/hal-04928888)
- Blangeois, M. (2025). L'Encycloscope : un assistant conversationnel pour explorer l'Encyclopédie de Diderot et d'Alembert. *EncyclopedIA*. https://encyclopedia.hypotheses.org/270

## Auteur

**Morgan Blangeois**
Doctorant en sciences de gestion, Clermont Recherche Management (CleRMa), Université Clermont Auvergne.
[ORCID 0009-0006-9699-037X](https://orcid.org/0009-0006-9699-037X)

## Licence

MIT
