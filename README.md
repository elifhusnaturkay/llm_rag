# Regulation-Aware Chatbot for Fırat University (RAG-based)

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) chatbot developed as part of a university software engineering internship project at the Fırat University Information Technology Department.

The chatbot is designed to assist users in querying institutional academic regulations (such as graduation requirements, credit systems, or course policies) by retrieving relevant segments from university documents and generating natural language answers. All operations are conducted locally using open-source tools and models.

## Project Scope

- Internship Period: Summer 2025  
- Organization: Fırat University, IT Department  
- Developer: Elif Hüsna Türkay, Software Engineering Student  
- Project Type: Academic Information Assistant (Offline, LLM-integrated)  

## Key Features

- Fully offline, local-only architecture (no external APIs or cloud dependencies)
- Document ingestion from structured academic texts in `.pdf`, `.docx`, or `.txt` format
- Embedding of text chunks using Ollama-powered models (e.g., `mxbai-embed-large`)
- Semantic search using ChromaDB
- Answer generation with local LLMs (e.g., `deepseek-llm:7b`) via Ollama
- Conversational memory: previous questions and answers are remembered and influence future responses

## Technologies Used

- Python 3.11
- Ollama (for both embedding and language models)
- ChromaDB (vector database)
- LangChain (for recursive character-based text chunking)
- PyPDF2, python-docx (for document parsing)

## File Structure
project_root/
├── pdf_files/ # Source document directory
│ └── egitim_ogrenci_isleri/ # Example topic folder with .txt files
├── chromadb_data/ # Automatically created vector DB
├── utils.py # Embedding pipeline: load, split, embed
├── ask.py # Chat interface with session memory
└── requirements.txt # All required packages
