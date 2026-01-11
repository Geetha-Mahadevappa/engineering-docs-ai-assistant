# engineering-docs-ai-assistant
## **Tech Stack**
- LangChain — https://python.langchain.com
- GraphQL — https://strawberry.rocks
- PyTorch FSDP — https://pytorch.org/docs/stable/fsdp.html
- FAISS — https://github.com/facebookresearch/faiss
- Model Context Protocol — https://modelcontextprotocol.io


## **Overview**

This project is a **small internal AI assistant** built to help **engineering teams** work with their **technical documents** more easily. It can search across **design docs, RFCs, runbooks, and notes,** and it brings everything together through a single agent that knows how to use different tools.

The system follows a **modular, MCP‑inspired design**. It exposes a **GraphQL API** for interacting with the assistant, uses **LangChain** to handle tool‑use and reasoning, and includes a **distributed‑ready fine‑tuning** setup powered by **PyTorch FSDP**. The goal is to provide a simple but realistic example of how modern AI systems organize document search, retrieval, and agentic workflows in a clean, production‑style way.

Here’s a short, clean, human‑written version you can drop straight into your README.  
It includes the link, why you use the dataset, and a quick note on its limitations — all in simple language.

---

## **Dataset**
This project uses the **StackSample** dataset from Kaggle:  
**`https://www.kaggle.com/datasets/stackoverflow/stacksample` [(kaggle.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fstackoverflow%2Fstacksample")**

It’s a small collection of real StackOverflow questions and answers. We use it only for fine‑tuning the embedding model because it gives us natural examples of how developers ask technical questions and how those questions are answered. This helps the model learn technical language before we apply it to real engineering documents.

The dataset is useful, but it’s not perfect. The text can be noisy, informal, and doesn’t fully match structured documents like RFCs or design specs. Because of that, we use it only for training and evaluation—not in production.
