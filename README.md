# 🧠 Engineering Docs AI Assistant
### **Secure • Multilingual • Hybrid Retrieval • Multi-Agent Orchestration**

![Python](img.shields.io)
![Retrieval](img.shields.io)
![Security](img.shields.io)
![LLM](img.shields.io)

---

## **🚀 Overview**
This project is a **production-ready AI assistant** designed for engineering teams to interact with sensitive technical documentation (RFCs, Design Docs, Runbooks). Unlike generic RAG systems, this assistant is built for **Data Sovereignty**, running entirely on local infrastructure to prevent internal data leakage to external APIs.

The system uses a modular, **Multi-Agent Architecture** where specialized agents handle document ingestion, conversation history, and high-precision hybrid retrieval.

---

## **🛠 Tech Stack**
*   **Orchestration:** [LangChain](https://python.langchain.com) (Tool-calling & Agentic Reasoning)
*   **Brain (LLM):** Local Llama 3.3 via [Ollama](https://ollama.com) or [vLLM](github.com)
*   **Document Memory:** [Qdrant](https://qdrant.tech) (Dense Vectors) & [Elasticsearch](https://www.elastic.co) (Keyword/BM25)
*   **Conversation Memory:** [Redis](https://redis.io) (Persistent Session Store)
*   **Ranking:** Local **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`)
*   **Inference:** Fine-tuned **Multilingual E5-base** (Stage-2 Hard-Negative training)
*   **API:** [GraphQL (Strawberry)](https://strawberry.rocks)

---

## 📚 Dataset (Embedding Fine-Tuning)

The embedding model is fine-tuned using the **StackSample** dataset from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stacksample), which is a small subset of real StackOverflow questions and answers.

Why this dataset?
- Written by real developers
- Noisy and informal, similar to real engineering queries
- Well-suited for learning technical question–answer relationships

The dataset is **used only for embedding fine-tuning** and is **not** used as a production knowledge source.

---

## 🔍 Embedding Fine-Tuning

We fine-tune the **intfloat/multilingual-e5-base** model to improve retrieval quality for engineering-focused content.

### Why fine-tune?

General-purpose embedding models often struggle with:
- Short technical questions
- Informal developer language
- Domain-specific terminology

Fine-tuning helps the model:
- Better understand developer-written queries
- Match questions to the *most relevant* technical documents
- Improve retrieval precision for internal documentation

### Training Strategy

Fine-tuning is performed in two stages:

#### Stage 1 — Supervised Training
- Train on **(question, correct answer)** pairs from StackOverflow
- Learn basic semantic alignment between questions and answers
- Produce an initial retrieval-capable embedding model

#### Stage 2 — Hard Negative Training
- Retrieve similar but incorrect answers using the Stage-1 model
- Train the model to distinguish correct answers from close distractors
- Sharpen the embedding space for higher retrieval precision

---

### 📊 Evaluation

The fine-tuned embedding model is evaluated on a **held-out split** of the StackSample dataset.

The evaluation set contains unseen **(query, answer)** pairs that were **not used** during either Stage-1 or Stage-2 training.  
For each query, we retrieve the top-k candidate answers from a **FAISS index** built over all answers and measure retrieval quality.

### Metrics

- **Recall@10** — How often the correct answer appears in the top-10 retrieved results  
- **MRR@10** — Mean Reciprocal Rank of the correct answer within the top-10  

### Results (Two-Stage Fine-Tuning)

| Model | Recall@10 | MRR@10 |
|------|-----------|--------|
| Base (no fine-tuning) | ~0.55 | ~0.32 |
| Stage-1 fine-tuned | ~0.70 | ~0.45 |
| Stage-2 (hard negatives) | **~0.78–0.82** | **~0.52–0.56** |

These results show that **hard-negative training significantly improves retrieval precision**, which directly benefits downstream document search and question-answering performance.

---

## **🏗 Agentic Architecture**

The system is organized into specialized agents that collaborate to ensure technical accuracy and data isolation.

### **1. Document Ingestion Agent**
Handles the transformation of unstructured files into searchable knowledge.
- **Formats:** PDF, Markdown, Plain Text.
- **Hybrid Indexing:** Generates **Dense E5 embeddings** for semantic meaning and **Sparse BM25 tokens** for exact keyword precision.
- **Batch Processing:** Handles large documents efficiently using configurable batch upserts to Qdrant.
- **Security:** Implements "Data Residue Protection," wiping raw files from server memory immediately after vectorization.

### **2. Retrieval Agent**
A two-stage high-precision engine that finds the "ground truth" for the assistant.
- **Stage 1 (Hybrid Recall):** Parallel search across Qdrant and Elasticsearch using **RRF (Reciprocal Rank Fusion)** to prioritize documents with multi-signal support.
- **Stage 2 (Reranking):** Uses a **Local Cross-Encoder** to validate candidates, ensuring technical details like error codes or specific function names match the query exactly.
- **Security Filters:** Enforces mandatory project-level and user-level isolation during every search.

### **3. Conversation Memory Agent**
Manages the short-term state and session persistence.
- **Redis-Backed:** History survives server restarts or agent crashes.
- **Resilient Fallback:** Proactive health checks automatically switch to in-process memory if Redis is unavailable.
- **Windowing:** Maintains a sliding window of the last 20 messages to optimize LLM context and performance.

### **4. Main Orchestrator Agent**
Uses a local **Llama** model to reason through user requests.
- **Query De-contextualization:** Uses conversation history to resolve follow-up questions (e.g., "Show me the Python example for *that*").
- **Multilingual Support:** Naturally understands and responds in the user's query language (English, German, etc.).
- **Resource Management:** Managed via a **TTLCache** to prevent RAM bloat while maintaining high-speed response times.

---

# WORK IN PROGRESS
