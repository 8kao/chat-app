# Context-Aware Chatbot

### Chatbot RAG (Retrieval-Augmented Generation) utilisant :

- ChromaDB

- OpenAI embeddings

- Chunking simple

- Embedding cache

- Conversation memory

- Query decomposition (multi-step RAG)

- Dynamic context filtering

- Conversation-aware prompting

---

## Fonctionnalités

### 1. Ingestion de documents

- Lecture automatique des fichiers dans data/docs_corpus

- Chunking simple (chunk_size=500, overlap=100)

- Embeddings générés par batch (OpenAI)

- Insertion en batch dans ChromaDB (pour éviter les erreurs de taille)


### 2. Embedding Cache

- Les embeddings des questions sont mis en cache -> si une question revient : pas de nouvel appel OpenAI


### 3. Retrieval basé sur la similarité

- Embedding de la requête

- Similarité cosinus

- Récupération des chunks les plus proches

- Filtrage par distance

- Déduplication par fichier

- Réduction à k documents les plus pertinents


### 4. Query Decomposition (Multi-step RAG)

Inspiré des techniques utilisées par Anthropic.

Quand l’utilisateur pose une question complexe comme :

"Explique-moi geometry ET compare avec physics"

Le système :

- Décompose la requête en sous-questions

- Fait un retrieval indépendant pour chaque sous-question

- Construit un prompt multi-contexte structuré

- Demande au LLM de fusionner les informations


### 5. Conversation Memory

L’historique est incorporé dans le prompt -> permet de garder le fil d’une conversation longue
Tous les 10 messages, l'historique de la conversation est envoyé à un autre LLM pour résumer et ne garder que les informations
importantes afin d'alléger le prompt donné à l'assistant.

---

## Installation 

### 1. Installation des dépendances

`pip install -r requirements.txt`

### 2. Ajouter la clé API OpenAI

Créer `.env` :

`cp .env.example .env`

`OPENAI_API_KEY="votre_cle"`

---

## Ingestion

Places tes docs ou garde ceux déjà en place dans `data/docs_corpus`

Lance :

`python source/embed.py`

---

## Utilisation de l'app de chat

Lance :

`python source/chat.py`