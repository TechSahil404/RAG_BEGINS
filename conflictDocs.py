import os
import tempfile
import streamlit as st
from dotenv import load_dotenv


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

# OCR imports
from pdf2image import convert_from_path
import pytesseract

# Load environment variables
load_dotenv()
 
st.set_page_config(page_title="📄 RAG System", layout="wide")
st.title("📄 Chat With Documents — RAG SYSTEM (Supports Text + Scanned PDFs)")

st.sidebar.header("Configuration")
st.sidebar.write(
    "- Enter your Groq API key\n"
    "- Upload PDF(s) (text or scanned)\n"
    "- Ask questions about the document(s)"
)

# API Key
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

if not api_key:
    st.warning("Please enter your Groq API key in sidebar or set GROQ_API_KEY in .env")
    st.stop()


llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")
# HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


uploaded_files = st.file_uploader("Upload PDF file(s)", type="pdf", accept_multiple_files=True)
all_docs = []

def extract_text_from_scanned_pdf(pdf_path):
    """Convert scanned PDF to text using OCR"""
    pages = convert_from_path(pdf_path)
    full_text = ""
    for page in pages:
        text = pytesseract.image_to_string(page)
        full_text += text + "\n"
    return full_text

if uploaded_files:
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # If no text extracted, try OCR
        if all([not d.page_content.strip() for d in docs]):
            ocr_text = extract_text_from_scanned_pdf(tmp_path)
            if ocr_text.strip():
                from langchain.schema import Document
                docs = [Document(page_content=ocr_text)]
        
        all_docs.extend(docs)

    st.success(f"Loaded {len(all_docs)} pages from {len(uploaded_files)} document(s).")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(all_docs)
    st.info(f"Total chunks created: {len(chunks)}")

    
    persist_dir = "chroma_db"
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    st.success("Vector store ready.")

    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    
    user_question = st.text_input("Ask a question about the document(s):")

    if st.button("Get Answer") and user_question.strip():
        result = qa_chain({"question": user_question, "chat_history": memory.chat_memory})
        answer = result.get("answer")
        source_docs = result.get("source_documents", [])

        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander("Show Source Documents"):
            if not source_docs:
                st.write("No source documents returned.")
            for i, doc in enumerate(source_docs, start=1):
                st.write(f"--- Document chunk {i} ---")
                st.write(doc.page_content[:1000])

    st.write("---")
    st.write("**Embedding model in use:** `sentence-transformers/all-MiniLM-L6-v2`")
