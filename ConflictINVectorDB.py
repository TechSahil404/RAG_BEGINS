import os
import tempfile
import streamlit as st
from dotenv import load_dotenv


from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory


load_dotenv()


st.set_page_config(page_title="📄 Chat With Documents", layout="wide")
st.title("📄 Chat With Documents — RAG SYSTEM (Strict Mode )")

st.sidebar.header("Configuration")
st.sidebar.write(
    "- Enter your Groq API key\n"
    "- Upload PDF(s)\n"
    "- Ask questions about the documents"
)


api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    st.warning("⚠️ Please enter your Groq API key in sidebar or set GROQ_API_KEY in .env")
    st.stop()


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  
)

llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")


uploaded_files = st.file_uploader("📎 Upload PDF file(s)", type="pdf", accept_multiple_files=True)
all_docs = []

if uploaded_files:
    # Load PDFs
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split()
        all_docs.extend(docs)

    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} document(s).")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(all_docs)
    st.info(f"Total chunks created: {len(chunks)}")

    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectordb.persist()
    st.success("Vector store ready.")

    
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    
    STRICT_PROMPT = """You are a strict document-based assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, reply exactly:
"No information found in the documents."
"""

    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    
    user_question = st.text_input("💬 Ask a question about the document(s):")

    if st.button("Get Answer") and user_question.strip():
        
        strict_question = f"{STRICT_PROMPT}\nQuestion: {user_question}"
        result = qa_chain({"question": strict_question, "chat_history": memory.chat_memory})
        answer = result.get("answer")
        source_docs = result.get("source_documents", [])

        st.markdown("### 📝 **Answer:**")
        st.write(answer)

        with st.expander("📚 Show Source Documents"):
            if not source_docs:
                st.write("No source documents returned.")
            for i, doc in enumerate(source_docs, start=1):
                st.write(f"--- Document chunk {i} ---")
                st.write(doc.page_content[:1000])

    st.write("---")
    st.caption("⚡ Embedding model: `sentence-transformers/all-MiniLM-L6-v2` | Strict Mode Enabled ✅")
