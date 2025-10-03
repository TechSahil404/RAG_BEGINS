A Retrieval‑Augmented Generation (RAG) system built with LangChain that lets users chat with their PDFs. The pipeline performs data loading (with an OCR-friendly path), chunking, embedding, vector indexing with Chroma, retrieval, and strictly grounded answer generation via a Groq chat model. The design highlights clean session isolation, reproducible retrieval, and guardrails against hallucinations.

Note: final.py is the working application. Other scripts are practice artifacts and not required to run the app.

## Why this project

- RAG-first: Demonstrates the canonical RAG loop (load → split → embed → index → retrieve → generate) using LangChain abstractions end‑to‑end.  
- Strict grounding: A system prompt enforces that answers come only from retrieved context; if missing, the model must reply “No information found in the documents.”  
- Practical ops: Per‑session Chroma collections avoid stale mixing; background cleanup prevents local store bloat; environment variables keep secrets out of source control.

## Architecture

1) Ingestion (Indexing)
- Load: PDFs via PyPDFLoader; if RapidOCR parser is present, images get parsed to text for stronger coverage.  
- Split: RecursiveCharacterTextSplitter creates overlapping chunks for better recall and context assembly.  
- Embed + Store: sentence‑transformers/all‑MiniLM‑L6‑v2 → Chroma vector store. Each app run writes to a unique persist directory and collection.

2) Retrieval + Generation (Online path)
- Retrieve: Top‑k (k=4 by default) similarity search over the Chroma collection scoped to the current session.  
- Generate: ChatGroq (gemma2‑9b‑it) receives the user query and retrieved context through a strict RAG prompt constructed with LangChain’s ChatPromptTemplate and chains.

3) Memory
- ConversationBufferMemory retains message history; a history‑aware retriever condenses follow‑up questions into standalone queries for stable retrieval.

## Core LangChain Components (in final.py)

- Document loaders: PyPDFLoader (+ optional RapidOCRBlobParser).  
- Text splitter: RecursiveCharacterTextSplitter (chunk_size=500, overlap=200).  
- Embeddings: HuggingFaceEmbeddings (all‑MiniLM‑L6‑v2).  
- Vector store: Chroma (session‑scoped persist directory and collection).  
- Retriever: Vector store retriever with k=4.  
- LLM: ChatGroq with model gemma2‑9b‑it.  
- Chains:
  - create_history_aware_retriever for query condensation.  
  - create_stuff_documents_chain for answer generation over retrieved docs.  
  - create_retrieval_chain wiring together retrieval → generation.

## Strict Prompting Policy

System prompt enforces:
- Use only provided context; do not fabricate.  
- If not present, respond exactly: “No information found in the documents.”

This ensures faithful, document‑grounded outputs.

## Project Layout

- final.py — Working app entry point (Streamlit).  
- requirements.txt — Project dependencies.  
- chroma_sessions/ — Auto‑created per run; holds the session’s embeddings index.  
- .env.example — Template listing required environment variables.  
- .env — Local‑only secrets (ignored by Git).  
- Other .py files — Practice code; not required to run the app.

## Prerequisites

- Python 3.10+ (3.11 recommended).  
- A Groq API key (GROQ_API_KEY).  
- Optional: HF_TOKEN for Hugging Face model downloads/rate limits.  
- Optional: RapidOCR parser installed if image‑heavy PDFs are common.

## Setup

1) Clone and enter
- HTTPS:
  git clone https://github.com/TechSahil404/RAG_BEGINS.git
  cd RAG_BEGINS

2) Create and activate venv (Windows)
- PowerShell:
  python -m venv .venv
  .\.venv\Scripts\activate

3) Install dependencies
  pip install -r requirements.txt

4) Configure environment
- Create a .env from the example:
  - Windows: copy .env.example .env
  - macOS/Linux: cp .env.example .env
- Open .env and set values:
  GROQ_API_KEY=your-groq-key
  HF_TOKEN=your-hf-token-optional

Note: .env is intentionally ignored by Git; only .env.example is committed.

## Run

- Streamlit:
  streamlit run final.py

In the sidebar:
- Paste Groq API key (or rely on GROQ_API_KEY from .env).  
- Upload one or more PDFs.  
- Ask questions.

## How final.py works (annotated flow)

- load_dotenv(): reads secrets from local .env.  
- Sidebar captures a masked API key; fallback is GROQ_API_KEY env var.  
- PDFs are uploaded and written to temporary files for loading.  
- Loader: PyPDFLoader uses extract_images=True; if RapidOCR parser is available, images get text extraction to boost recall on scanned docs.  
- Each Document gets a “source” metadata set to the original filename to aid traceability.  
- Chunks created via RecursiveCharacterTextSplitter for balanced recall and context length.  
- Chroma.from_documents() builds a fresh, session‑scoped index and persists it.  
- Retriever is constructed against the specific session collection.  
- History‑aware retriever condenses follow‑ups; RAG chain answers via strict prompt.  
- Source chunks are displayed so users can audit the grounding.

## Tuning and Extensions

- k (top‑k retrieval): Increase for more context, decrease for speed.  
- Embeddings model: Swap to stronger models (e.g., bge) if better semantic recall is needed.  
- Vector store: Replace Chroma with a hosted store for multi‑user deployments.  
- OCR: Add robust OCR pipeline (e.g., Unstructured + Tesseract/RapidOCR) for scanned PDFs.  
- Prompting: Add citations, bullet constraints, or JSON output formatting in the system prompt.  
- Memory: Persist conversation history for session continuity.

