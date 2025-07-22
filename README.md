# Regulation-Aware RAG Chatbot – Fırat University Internship Project

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) chatbot developed during a summer internship at the Fırat University IT Department.

The chatbot is designed to answer user questions about Fırat University’s academic regulations by retrieving relevant document segments and generating context-aware responses. The system operates entirely offline, leveraging Ollama for both embedding and language model inference.

---

## System Architecture

- **Embedding Model**: `nomic-embed-text` (via Ollama)  
- **Language Model**: `gemma3:12b-it` (via Ollama)  
- **Database**: ChromaDB (local vector store)  
- **Chunking**: 1000-character segments with 100-character overlap using LangChain

---

## Improvements Over Previous Versions

- **Contextual Consistency**:  
  Longer chunks with strategic overlap eliminated the fragmenting problem previously observed in retrieval.  
  This led to a significant improvement in semantic relevance during inference.

- **Enhanced Embedding Quality**:  
  The `nomic-embed-text` model was adopted due to its multilingual capabilities and strong support for Turkish. It replaced former Hugging Face-based models to ensure a 100% Ollama-based, offline architecture.

- **Upgraded Language Model**:  
  The `gemma3:12b-it` model was selected for its superior performance on instruction-following tasks and strong Turkish language understanding.

---


## Running the Project

### Step 1 – Pull Required Ollama Models
```bash
ollama pull nomic-embed-text
ollama pull gemma3:12b-it
pip install -r requirements.txt
python utils.py
python ask.py



