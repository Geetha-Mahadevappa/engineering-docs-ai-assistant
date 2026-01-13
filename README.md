# engineering-docs-ai-assistant
## **Tech Stack**
- [LangChain](https://python.langchain.com)
- [GraphQL (Strawberry)](https://strawberry.rocks)
- [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Model Context Protocol](https://modelcontextprotocol.io)



## **Overview**

This project is a **small internal AI assistant** built to help **engineering teams** work with their **technical documents** more easily. It can search across **design docs, RFCs, runbooks, and notes,** and it brings everything together through a single agent that knows how to use different tools.

The system follows a **modular, MCP‑inspired design**. It exposes a **GraphQL API** for interacting with the assistant, uses **LangChain** to handle tool‑use and reasoning, and includes a **distributed‑ready fine‑tuning** setup powered by **PyTorch FSDP**. The goal is to provide a simple but realistic example of how modern AI systems organize document search, retrieval, and agentic workflows in a clean, production‑style way.

---

## **Dataset**
This project uses the **StackSample** dataset from Kaggle: [(kaggle.com)](https://www.kaggle.com/datasets/stackoverflow/stacksample)

It’s a small subset of real StackOverflow questions and answers. It provides natural developer‑written queries and responses, making it useful for training and evaluating the embedding model.
The dataset is noisy and informal, so it’s used only for fine‑tuning—not as a production knowledge source.

## **Architecture**
This project uses a modular, MCP‑inspired architecture to power an internal AI assistant for engineering teams. The system is built in stages, and the first foundational stage is **embedding fine‑tuning**, which creates the retrieval backbone for the entire assistant.

### **1. Embedding Fine‑Tuning**
We fine‑tune **intfloat/multilingual-e5-base**, a strong embedding model that supports multiple languages, including English and German.
This makes it a natural fit for engineering teams that work across bilingual documentation or mixed‑language notes.

Fine‑tuning adapts the model to developer‑written technical content so it retrieves design docs, RFCs, and runbooks with higher accuracy.

**The training happens in two stages:**
#### **Stage 1 — Supervised Training**
- Build (query, positive) pairs from StackOverflow
- Apply E5 prefixes (query: / passage:)
- Train with MNLR loss
- Build a Stage‑1 FAISS index

This gives the model a basic understanding of how technical questions map to correct answers.

#### **Stage 2 — Hard‑Negative Training**
- Use the Stage‑1 model to retrieve similar but incorrect answers
- Select the closest incorrect answer as the hard negative
- Train on (query, positive, hard_negative)
- Build the final FAISS index

This sharpens the embedding space and significantly boosts retrieval precision.

![Fine‑Tuning Pipeline]()<img width="1013" height="567" alt="Screenshot from 2026-01-12 15-15-11" src="https://github.com/user-attachments/assets/4a884ca3-e2d5-4c5a-ab97-1a557a267cb5" />

## **Evaluation**

We evaluate the fine‑tuned embedding model on a **held‑out split** of the StackSample dataset.  
The evaluation set contains unseen `(query, answer)` pairs that were **not** used during Stage‑1 or Stage‑2 training.  
For each question, we retrieve the top‑k answers from a FAISS index built over all candidate answers.

**Metrics**
- **Recall@10** — how often the correct answer appears in the top‑10  
- **MRR@10** — how high the correct answer ranks on average  

**Typical results for two‑stage fine‑tuning**

| Model | Recall@10 | MRR@10 |
|-------|-----------|--------|
| Base (no fine‑tuning) | ~0.55 | ~0.32 |
| Stage‑1 fine‑tuned | ~0.70 | ~0.45 |
| Stage‑2 (hard negatives) | **~0.78–0.82** | **~0.52–0.56** |

Two‑stage training consistently improves retrieval quality, especially in ranking similar but incorrect answers lower.

#### **How We Can Improve Further**

Even though two‑stage fine‑tuning provides strong retrieval performance, several upgrades can push the system closer to production‑grade quality:

**1. Cross‑Encoder Reranking**

Bi‑encoder retrieval (FAISS) is fast but loses fine‑grained interactions.  
A reranker processes the query and document together and catches details like function names or error codes.

**Workflow:**  

1. Use the Stage‑2 E5 model to retrieve top‑50 candidates.  
2. Rerank those 50 using a Cross‑Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).  

This typically gives the **largest single boost in MRR** after hard‑negative training.

---

**2. Domain‑Specific Prefixes**

Generic `query:` and `passage:` prefixes are broad.  
Engineering‑focused prefixes improve embedding alignment.

Examples:  
- **Search prefix:** “Represent the engineering query for retrieving relevant documentation:”  
- **Index prefix:** “Represent the technical documentation for retrieval:”  

Make these configurable in `embedded_training.yaml` and pass them through your trainer.

---

**3. Better Hard‑Negative Mining (Batch‑Hard)**
Instead of mining negatives once before Stage‑2, use **in‑batch hard negatives**:

- For each query, find the most similar *incorrect* sample in the batch.  
- Apply higher loss weight to that specific negative.  

This forces the model to learn sharper decision boundaries.

---

**4. Synthetic Data (LLM‑in‑the‑Loop)**
If you lack real `(query, positive)` pairs from engineering docs:

1. Take a design doc or RFC.  
2. Use an LLM to generate 3–5 realistic engineering questions about it.  
3. Add these to Stage‑1 training.  

This injects domain vocabulary and improves retrieval on internal documents.

---

These improvements can be added incrementally.  
The **Cross‑Encoder reranker** is the highest‑impact next step for boosting ranking quality.

---


