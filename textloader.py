from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import streamlit as st

# ------------------------------------------------------------------------------------------------------------------
# Load documents (example text + pdf)
Loader = TextLoader("example2.txt", encoding="utf-8")
docs = Loader.load()

pdf_loader = UnstructuredPDFLoader("attention.pdf", mode="elements")
pdf_docs = pdf_loader.load()

# Merge both
docs.extend(pdf_docs)

# -----------------------------------------------------------------------------------------------------------------
# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    separators=["\n", " ", ""]
)

chunks = text_splitter.split_documents(docs)

# ------------------------------------------------------------------------------------------------------------------
# Embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)

# -------------------------------------------------------------------------------------------------------------------
# LLM setup (CPU-friendly)
model_name = "google/flan-t5-small"  # small CPU-friendly model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------------------------------------------------------------------------------------------
# Streamlit App
st.set_page_config(page_title="Simple RAG Demo", layout="centered")
st.title("📚 Simple RAG Demo")

query = st.text_input("Enter your question:", key="main_query")

if st.button("Submit"):
    if query.strip():
        st.success(f"You asked: {query}")

        # Retrieve top 3 relevant chunks
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Format prompt and get answer
        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        answer = llm(prompt)

        st.write("💡 Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question before submitting.")
