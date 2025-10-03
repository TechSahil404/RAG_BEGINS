from langchain_community.document_loaders import TextLoader  # community package jisme sare loaders hote h 
from langchain_text_splitters import RecursiveCharacterTextSplitter  # text splitters ke liye 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import UnstructuredPDFLoader  # pdf loader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import streamlit as st
import tempfile

#------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Simple RAG Demo", layout="centered")
st.title("📚 Simple RAG Demo")

# File upload option
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
uploaded_txt = st.file_uploader("Upload a TXT file", type=["txt"])

docs = []

#------------------------------------------------------------------------------------------------------------------
# Load TXT if provided
if uploaded_txt is not None:
    Loader = TextLoader(uploaded_txt, encoding="utf-8")
    docs = Loader.load()
    st.success(f"Loaded {len(docs)} document(s) from TXT file")

# Load PDF if provided
elif uploaded_pdf is not None:
    # Save PDF temporarily so loader can access a real file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    Loader = UnstructuredPDFLoader(tmp_path, mode="elements")
    docs = Loader.load()
    st.success(f"Loaded {len(docs)} page(s) from PDF file")

#------------------------------------------------------------------------------------------------------------------
if docs:
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        separators=["\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    st.info(f"Total chunks created: {len(chunks)}")

    # Embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    chunk_embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])

    # Vector store
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)

    #------------------------------------------------------------------------------------------------------------------
    # User question
    query = st.text_input("Enter your question:", key="app_input")

    if st.button("Submit"):
        if query.strip():
            st.success(f"You asked: {query}")

            # Retrieve top 3 relevant chunks
            retrieved_docs = vector_store.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            # LLM setup
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200
            )
            llm = HuggingFacePipeline(pipeline=pipe)

            # Prompt & Answer
            prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
            answer = llm(prompt)
            st.write("💡 Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question before submitting.")
else:
    st.info("Please upload a PDF or TXT file to start.")
