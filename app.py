import os
import streamlit as st
import google.generativeai as genai

import pdfplumber
import camelot
from PIL import Image
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# Gemini Setup
# -------------------------
API_KEY = "AIzaSyBIiJbgGDi29cAtJhpJW9wc6DX98IiLo4s"  # 👈 apna key daal
genai.configure(api_key=API_KEY)

EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"


# -------------------------
# Extractors
# -------------------------
def extract_text_from_pdf(file_path):
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return texts


def extract_tables_from_pdf(file_path):
    tables = []
    try:
        t = camelot.read_pdf(file_path, pages="all")
        for table in t:
            tables.append(table.df.to_string())
    except Exception as e:
        print("Table extraction error:", e)
    return tables


def extract_text_from_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return [text] if text else []


# -------------------------
# Gemini Embedding
# -------------------------
def embed_texts_gemini(texts, embedding_model=EMBED_MODEL):
    embeddings = []
    for t in texts:
        if not t.strip():
            continue
        resp = genai.embed_content(model=embedding_model, content=t)
        emb = resp["embedding"]
        embeddings.append((t, emb))
    return embeddings


# -------------------------
# Build Vector DB
# -------------------------
def build_vectorstore(texts):
    pairs = embed_texts_gemini(texts)

    # LangChain FAISS store (dummy HuggingFace embeddings for structure)
    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = [Document(page_content=t[0]) for t in pairs]
    vectors = FAISS.from_documents(docs, emb_model)
    return vectors


# -------------------------
# Gemini QA
# -------------------------
def gemini_answer(context, query):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel(LLM_MODEL)
    resp = model.generate_content(prompt)
    return resp.text


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Universal RAG System", layout="wide")

st.title("📚 Universal RAG System (Text + Table + Image) with Gemini")

uploaded_file = st.file_uploader("Upload a PDF/Image/Text file", type=["pdf", "png", "jpg", "jpeg", "txt"])

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    texts = []
    if uploaded_file.type == "application/pdf":
        texts.extend(extract_text_from_pdf(file_path))
        texts.extend(extract_tables_from_pdf(file_path))
    elif "image" in uploaded_file.type:
        texts.extend(extract_text_from_image(file_path))
    elif uploaded_file.type == "text/plain":
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    st.success(f"Extracted {len(texts)} chunks of text/tables/images ✅")

    if texts:
        # Split texts
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_texts = []
        for t in texts:
            split_texts.extend(splitter.split_text(t))

        # Build vector store
        vectorstore = build_vectorstore(split_texts)

        query = st.text_input("🔍 Ask a question based on uploaded file:")
        if query:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs])

            answer = gemini_answer(context, query)
            st.markdown("### 🤖 Answer:")
            st.write(answer)
