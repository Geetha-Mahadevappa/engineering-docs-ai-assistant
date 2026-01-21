# 🧠 Engineering Docs AI Assistant
### **Secure • Multilingual • Hybrid Retrieval • Multi-Agent Orchestration**

![Python](img.shields.io)
![Retrieval](img.shields.io)
![Security](img.shields.io)
![LLM](img.shields.io)

---

## 🚀 Overview

This project is a **production-ready AI assistant** designed for engineering teams to interact with sensitive technical documentation (RFCs, Design Docs, Runbooks). Unlike generic RAG systems, this assistant is built for **Data Sovereignty**, running entirely on local infrastructure to prevent internal data leakage to external APIs.

The system uses a modular **multi-agent architecture**, where specialized agents handle document ingestion, conversation memory, and high-precision hybrid retrieval.

---

## 🛠 Tech Stack

- **Orchestration:** LangChain (tool-calling & agentic reasoning)
- **Brain (LLM):** Local Llama 3.3 via Ollama or vLLM
- **Document Memory:** Qdrant (dense vectors) + Elasticsearch (BM25)
- **Conversation Memory:** Redis (persistent sessions)
- **Ranking:** Local cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- **Embeddings:** Fine-tuned multilingual E5-base (hard-negative training)
- **API:** GraphQL (Strawberry)

---

## 📚 Dataset (Embedding Fine-Tuning)

The embedding model is fine-tuned using the **StackSample** dataset from Kaggle, a curated subset of real StackOverflow questions and answers.

The dataset is:
- Written by real developers
- Informal and noisy (similar to real engineering queries)
- Used **only for embedding fine-tuning**, not as a knowledge source

---

## 🔍 Embedding Fine-Tuning

We fine-tune **multilingual-e5-base** to improve retrieval quality for engineering-focused content.

### Training Strategy

**Stage 1 — Supervised Training**
- Train on (question, correct answer) pairs
- Learn basic semantic alignment

**Stage 2 — Hard Negative Training**
- Introduce semantically similar but incorrect answers
- Improve discrimination and retrieval precision

---

## 📊 Embedding Evaluation

Evaluation is performed on a **held-out split** of the StackSample dataset.

**Metrics**
- **Recall@10**
- **MRR@10**

| Model | Recall@10 | MRR@10 |
|------|-----------|--------|
| Base (no fine-tuning) | ~0.55 | ~0.32 |
| Stage-1 fine-tuned | ~0.70 | ~0.45 |
| Stage-2 (hard negatives) | **~0.78–0.82** | **~0.52–0.56** |

---

## 🏗 Agentic Architecture

### 1. Document Ingestion Agent
- Supports PDF, Markdown, and plain text
- Hybrid indexing (dense + BM25)
- Secure memory cleanup after ingestion

### 2. Retrieval Agent
- Hybrid recall with RRF (Qdrant + Elasticsearch)
- Cross-encoder reranking for precision
- Enforces project- and user-level isolation

### 3. Conversation Memory Agent
- Redis-backed persistent sessions
- Sliding window (last 20 messages)
- Fallback to in-process memory if Redis is unavailable

### 4. Main Orchestrator Agent
- Query de-contextualization
- Multilingual understanding
- Resource management via TTL caching

---

## 🧪 Evaluation & Quality Assurance

The system includes an **offline evaluation pipeline** located in `/eval`.

### Evaluation Setup

- **Evaluation Source:**  
  Kubernetes official documentation  
  https://kubernetes.io/docs/concepts/

- **Evaluation Queries:**  
  30 manually curated queries derived from the documentation

### Evaluation Agents

- **Evaluation Coordinator Agent** – runs evaluation jobs and aggregates results  
- **Ground-Truth Validator Agent** – checks semantic correctness  
- **Faithfulness Auditor Agent** – verifies source grounding  
- **Performance Monitor Agent** – measures latency, throughput, and errors  

### Metrics Tracked

| Metric | Description | Baseline |
|------|-------------|----------|
| Correctness | Semantic match with ground truth | 78–88% |
| Faithfulness | Answer supported by sources | 70–85% |
| Recall@10 | Correct doc in top-10 | 0.78–0.82 |
| MRR@10 | Rank quality of correct doc | 0.52–0.56 |
| Latency (cached) | End-to-end response time | < 1.2s |
| Latency (cold) | End-to-end response time | < 3.5s |
| Throughput | Sustained RPS | 50–200 |
| Error Rate | Failed requests | < 1% |

---

## 🖥 UI Overview

The UI focuses on **transparency, debugging, and trust**.

**Sections**
- Query bar
- Conversation pane
- Source panel with provenance
- Document browser
- Admin panel

**Features**
- Source highlighting & evidence view
- Multilingual support
- Session export
- User feedback (correct / incorrect)
- Visible project-level access control

*(Screenshots will be added.)*

---

## ⚙️ Build & Run

### Prerequisites

- Docker
- NVIDIA Container Toolkit (only if using GPU)
- `config.env` with endpoints and secrets:
  - Qdrant
  - Elasticsearch
  - Redis
  - Model server

### Build & Start

```bash
docker compose build
docker compose up
```
---

## ✅ Conclusion

This project presents a **secure, production-oriented AI assistant** for engineering documentation.  
Through **hybrid retrieval**, **multi-agent orchestration**, and **offline evaluation**, the system is designed to deliver accurate, grounded, and multilingual responses while preserving **data sovereignty**.

The architecture and evaluation setup make it well-suited for **enterprise and regulated environments**, where transparency, reliability, and system control are critical.

