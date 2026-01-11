# engineering-docs-ai-assistant
## Tech Stack
- LangChain — https://python.langchain.com
- GraphQL — https://strawberry.rocks
- PyTorch FSDP — https://pytorch.org/docs/stable/fsdp.html
- FAISS — https://github.com/facebookresearch/faiss
- Model Context Protocol — https://modelcontextprotocol.io


## Overview

This project is a **small internal AI assistant** built to help **engineering teams** work with their **technical documents** more easily. It can search across **design docs, RFCs, runbooks, and notes,** and it brings everything together through a single agent that knows how to use different tools.

The system follows a **modular, MCP‑inspired design**. It exposes a **GraphQL API** for interacting with the assistant, uses **LangChain** to handle tool‑use and reasoning, and includes a **distributed‑ready fine‑tuning** setup powered by **PyTorch FSDP**. The goal is to provide a simple but realistic example of how modern AI systems organize document search, retrieval, and agentic workflows in a clean, production‑style way.
