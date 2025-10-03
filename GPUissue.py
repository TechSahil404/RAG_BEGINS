import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ---------------- Safe Chroma import (no crash) ----------------
try:
    from langchain_chroma import Chroma  # preferred; pip install -U langchain-chroma chromadb
    _CHROMA_IMPORT = "new"
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback; pip install -U langchain-community chromadb
    _CHROMA_IMPORT = "community"

# ---------------- LangChain imports ----------------
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

# ---------------- Load .env ----------------
load_dotenv()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📄 Chat With Documents", layout="wide")
st.title("📄 Chat With Documents — RAG SYSTEM (Strict Mode ✅)")

st.sidebar.header("⚙️ Configuration")
st.sidebar.write(
    "- Enter your Groq API key\n"
    "- Upload PDF(s)\n"
    "- Ask questions about the documents"
)

# ---------------- API key handling ----------------
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    st.warning("⚠️ Please enter your Groq API key in sidebar or set GROQ_API_KEY in .env")
    st.stop()

# ---------------- Initialize Embeddings and LLM ----------------
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Force embeddings to CPU
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")

# ---------------- File Upload ----------------
uploaded_files = st.file_uploader("📎 Upload PDF file(s)", type="pdf", accept_multiple_files=True)
all_docs = []

if uploaded_files:
    # Load PDFs with OCR (important for image-based certificates)
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        # Try RapidOCR parser if available; fallback to basic
        try:
            # RapidOCR parser offers OCR for images inside PDF
            from langchain_community.document_loaders.parsers import RapidOCRBlobParser
            loader = PyPDFLoader(
                tmp_path,
                extract_images=True,
                images_inner_format="text",
                images_parser=RapidOCRBlobParser(),  # OCR engine
                mode="page",
            )
        except Exception:
            # Fallback (still extracts images; OCR depends on env)
            loader = PyPDFLoader(tmp_path, extract_images=True, mode="page")

        docs = loader.load()  # pages as documents
        all_docs.extend(docs)

    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} document(s).")

    # ---------------- Split documents ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(all_docs)
    st.info(f"🧠 Total chunks created: {len(chunks)}")

    # ---------------- Vector Store (create or load safely) ----------------
    PERSIST_DIR = "chroma_db"
    COLLECTION = "rag_strict_pdf"  # isolate from other projects

    if os.path.exists(PERSIST_DIR):
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION,
            embedding_function=embeddings
        )
        # If empty, build it
        if len(vectordb.get().get("ids", [])) == 0 and len(chunks) > 0:
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIR,
                collection_name=COLLECTION
            )
            vectordb.persist()
    else:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION
        )
        vectordb.persist()

    st.success("📌 Vector store ready.")

    # ---------------- Retriever (Strict) ----------------
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # ---------------- Memory ----------------
    memory = ConversationBufferMemory(return_messages=True)

    # ---------------- Strict Prompts ----------------
    STRICT_PROMPT = (
        "You are a strict document-based assistant. "
        "Use ONLY the provided context to answer the question. "
        "If the answer is not present in the context, reply exactly: "
        "\"No information found in the documents.\" \n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", STRICT_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the latest question as a standalone query using chat history. Do not answer."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # Build history-aware retriever + QA chain (recommended)
    history_aware = create_history_aware_retriever(llm, retriever, condense_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, qa_chain)

    # ---------------- User Query ----------------
    user_question = st.text_input("💬 Ask a question about the document(s):")

    if st.button("Get Answer") and user_question.strip():
        result = rag_chain.invoke({"input": user_question, "chat_history": memory.chat_memory.messages})
        answer = result.get("answer")
        context_docs = result.get("context", [])

        st.markdown("### 📝 **Answer:**")
        st.write(answer)

        with st.expander("📚 Show Source Documents"):
            if not context_docs:
                st.write("No source documents returned.")
            for i, doc in enumerate(context_docs, start=1):
                st.write(f"--- Document chunk {i} ---")
                st.write(doc.page_content[:1000])

        # update memory
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(answer)

    st.write("---")
    st.caption("⚡ Embedding model: `sentence-transformers/all-MiniLM-L6-v2` | Strict Mode Enabled ✅")
