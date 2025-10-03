import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain prompts + utilities
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate

# Groq LLM
from langchain_groq import ChatGroq

# Text splitting + embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Vectorstore
from langchain.vectorstores import Chroma

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Memory
from langchain.memory import ConversationBufferMemory

# Conversational retrieval chain
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Chat With Documents", layout="wide")
st.title("📄 Chat With Documents — RAG SYSTEM (Final Stable)")

st.sidebar.header("Configuration")
st.sidebar.write(
    "- Enter your Groq API key\n"
    "- Upload PDF(s)\n"
    "- Ask questions about documents"
)

# API key input
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar or set GROQ_API_KEY in .env")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")

# File uploader
uploaded_files = st.file_uploader("Upload PDF file(s)", type="pdf", accept_multiple_files=True)

all_docs = []

if uploaded_files:
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split()
        all_docs.extend(docs)

    st.success(f"Loaded {len(all_docs)} pages from {len(uploaded_files)} document(s).")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(all_docs)
    st.info(f"Total chunks created: {len(chunks)}")

    # Chroma vector store
    persist_directory = "chroma_db"
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    st.success("Vector store ready.")

    # Conversation memory (specify output_key for multi-output chains)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # tells memory to store only 'answer'
    )

    # Build chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    # User question input
    user_question = st.text_input("Ask a question about the document(s):")

    if st.button("Get Answer") and user_question.strip():
        # Pass question + chat_history
        result = qa_chain({
            "question": user_question,
            "chat_history": memory.chat_memory  # memory manages multi-turn conversation
        })

        answer = result.get("answer")
        source_docs = result.get("source_documents", [])

        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander("Show Source Documents"):
            if not source_docs:
                st.write("No source documents returned.")
            for i, doc in enumerate(source_docs, start=1):
                st.write(f"--- Document chunk {i} ---")
                st.write(doc.page_content[:1000])  # show first 1000 chars

    st.write("---")
    st.write("**Embedding model in use:** `sentence-transformers/all-MiniLM-L6-v2`")
