# LangChain-based Retrieval-Augmented Generation (RAG) for Q&A System

This repository implements a Q&A system using LangChain with document-based retrieval and text generation using a Hugging Face pipeline. The system allows you to query knowledge stored in text files and receive contextually accurate responses. This is achieved using a retrieval-augmented generation (RAG) approach where documents are split, indexed, embedded, and then retrieved to provide relevant context for the LLM to generate answers.

## Prerequisites

Before running this code, make sure to install the following Python libraries:

```bash
pip install langchain
pip install transformers
pip install faiss-cpu
pip install sentence-transformers
