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

Here’s a short, clean, human‑written version you can drop straight into your README.  
It includes the link, why you use the dataset, and a quick note on its limitations — all in simple language.

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

