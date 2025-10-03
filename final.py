import os
import shutil
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

# Chroma 
try:
    from langchain_chroma import Chroma 
except Exception:
    from langchain_community.vectorstores import Chroma  

# LangChain imports 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

# .env
load_dotenv()

# Streamlit UI 
st.set_page_config(page_title="📄 Chat With Documents", layout="wide")
st.title("📄 Chat With Documents — RAG SYSTEM (Strict Mode ✅)")

st.sidebar.header("⚙️ Configuration")
st.sidebar.write("- Enter your Groq API key\n- Upload PDF(s)\n- Ask questions about the documents")

# API key handling 
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password") or os.getenv("GROQ_API_KEY", "")
if not api_key:
    st.warning("⚠️ Please enter your Groq API key in sidebar or set GROQ_API_KEY in .env")
    st.stop()

# Initialize Embeddings and LLM 
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")

# Session-scoped Chroma config 
BASE_PERSIST_ROOT = "chroma_sessions"   # root for all sessions
SESSION_TS = str(int(time.time()))      # unique per run
PERSIST_DIR = os.path.join(BASE_PERSIST_ROOT, SESSION_TS)
COLLECTION = f"rag_pdf_{SESSION_TS}"

# Helper: background cleanup for older sessions (best-effort, Windows-safe)
def try_delete_old_sessions(root: str, max_age_hours: int = 12):
    if not os.path.isdir(root):
        return
    cutoff = time.time() - max_age_hours * 3600
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if path == PERSIST_DIR:
            continue
        # skip very recent or non-numeric names
        try:
            ts = int(name)
            if ts > cutoff:
                continue
        except Exception:
            continue
        # delete with backoff to bypass Windows file locks
        for _ in range(5):
            try:
                shutil.rmtree(path)
                break
            except PermissionError:
                time.sleep(1)

# Kick off cleanup (non-blocking effect)
try_delete_old_sessions(BASE_PERSIST_ROOT)

# ---------------- File Upload ----------------
uploaded_files = st.file_uploader("📎 Upload PDF file(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        # Prefer RapidOCR image-to-text if available
        try:
            from langchain_community.document_loaders.parsers import RapidOCRBlobParser
            loader = PyPDFLoader(
                tmp_path,
                extract_images=True,
                images_inner_format="text",
                images_parser=RapidOCRBlobParser(),
                mode="page",
            )
        except Exception:
            loader = PyPDFLoader(tmp_path, extract_images=True, mode="page")

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded.name
        all_docs.extend(docs)

    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} document(s).")

    # ---------------- Split documents ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(all_docs)
    st.info(f"🧠 Total chunks created: {len(chunks)}")

    # ---------------- Fresh Vector Store (isolated) ----------------
    # Create session-unique collection+directory to avoid mixing old docs
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )
    # Chroma >=0.4 auto-persists; calling persist() is optional.
    try:
        vectordb.persist()
    except Exception:
        pass

    st.success(f"📌 Vector store ready: {COLLECTION}")

    # ---------------- Retriever bound to this collection ----------------
    retriever = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    ).as_retriever(search_kwargs={"k": 4})

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

    # ---------------- RAG Chain ----------------
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

        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(answer)

    st.write("---")
    st.caption(f"⚡ Embedding model: `sentence-transformers/all-MiniLM-L6-v2` | Collection: {COLLECTION} | Created by SAHIL ")
