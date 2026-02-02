# Agentic Conversational RAG System for PDF Question Answering

**Personal Project**  
An advanced **Agentic Retrieval-Augmented Generation (RAG)** application that enables context-aware, multi-step question answering over user-uploaded PDF documents using autonomous reasoning agents.

---

## Overview

This project extends a traditional RAG pipeline into an **agent-based architecture**, where user queries are dynamically analyzed, decomposed, and resolved through coordinated retrieval and reasoning steps. The system is designed to generate **grounded, contextually accurate responses** by iteratively interacting with a vector database and large language models.

---

## Tech Stack

- **LLM Inference**: Groq API (Gemma-2B)
- **Agent & RAG Frameworks**: LangChain, LlamaIndex
- **Vector Store**: ChromaDB
- **Embeddings**: Hugging Face Sentence Transformers
- **Frontend**: Streamlit
- **Containerization**: Docker

---

## Architecture Highlights

### Agent-Oriented Query Handling
- Implemented autonomous agents to interpret user intent and decompose complex queries into smaller sub-tasks.
- Enabled multi-step reasoning where agents decide *what to retrieve*, *when to retrieve*, and *how to synthesize responses*.

### Retrieval-Augmented Generation Pipeline
- PDF documents are chunked and embedded using Hugging Face models.
- Semantic similarity search is performed via ChromaDB to retrieve relevant context.
- Retrieved context is iteratively refined and passed to the LLM for grounded response generation.

### Context-Aware Conversational Flow
- Maintains persistent chat history to support follow-up questions and conversational continuity.
- Improves answer relevance by leveraging both retrieved documents and prior interaction context.

### Production-Ready Design
- Modular codebase separating ingestion, retrieval, agent reasoning, and response generation.
- Fully containerized using Docker for portability and cloud deployment readiness.

---

## Key Features

- Agentic query decomposition and multi-step reasoning
- Semantic document retrieval over PDF knowledge bases
- Context-grounded answer generation with reduced hallucination
- Persistent conversational memory
- Dockerized setup for reproducible deployments

---

## Use Cases

- Interactive document exploration and knowledge extraction
- Technical document Q&A systems
- Enterprise or research-focused knowledge assistants

---

## Future Enhancements

- Multi-agent collaboration (planner–executor pattern)
- Tool-augmented agents (search, summarization, citation tracing)
- Streaming responses and latency optimization
- Cloud deployment with CI/CD integration

---

## Security Notes

All API keys and credentials are managed via environment variables and are excluded from version control to follow best security practices.
