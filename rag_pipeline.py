# rag_pipeline.py
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

import pytesseract
from pdf2image import convert_from_path
import pdfplumber

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ✅ HuggingFace embeddings (free & unlimited)
from langchain_huggingface import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings (model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# ---------- CONFIG ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set. Use setx or .env file.")

TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if not shutil.which("tesseract") and Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
# --------------------------------


def load_pdf(file_path: str):
    """
    Load PDF with unstructured (handles text/images/tables). 
    Fallback: pdfplumber + OCR if needed.
    """
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
    except Exception:
        documents = []

    # >> operation: document loader (UnstructuredPDFLoader attempted)

    if not documents:
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    texts.append(Document(page_content=text, metadata={"source": file_path, "page": page_no + 1}))
                else:
                    images = convert_from_path(file_path, first_page=page_no + 1, last_page=page_no + 1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        texts.append(Document(page_content=ocr_text, metadata={"source": file_path, "page": page_no + 1}))
        documents = texts

    # >> comment: document loader complete (unstructured or fallback OCR)
    return documents


def split_documents(documents):
    """
    Split loaded documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content or "")
        for i, s in enumerate(splits):
            md = dict(doc.metadata) if doc.metadata else {}
            md.update({"chunk": i})
            chunks.append(Document(page_content=s, metadata=md))
    # >> operation: texts split into chunks
    return chunks


def create_vector_db(docs_texts, persist_dir="chroma_db"):
    """
    Embed chunks with Gemini embeddings and store in Chroma DB.
    """
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(documents=docs_texts, embedding=embed_model, persist_directory=persist_dir)
    vectordb.persist()
    # >> operation: vector DB created and persisted
    return vectordb, embed_model


def get_llm():
    """
    Initialize Gemini chat LLM.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    # >> operation: Chat LLM initialized
    return llm


def answer_query(question, vectordb, llm, k=4):
    """
    Run RAG: retrieve relevant chunks and generate answer with LLM.
    """
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # >> operation: retriever created

    related_docs = retriever.get_relevant_documents(question)
    # >> operation: retriever fetched relevant chunks

    context_texts = "\n\n---\n\n".join([d.page_content for d in related_docs])
    prompt = (
        "You are an assistant that answers user questions using ONLY the provided document content. "
        "If the answer is not in the content, say you don't know. Provide concise, accurate answers.\n\n"
        f"DOCUMENT CONTEXT:\n{context_texts}\n\n"
        f"USER QUESTION: {question}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)
    # >> operation: LLM generated answer
    return response, related_docs
