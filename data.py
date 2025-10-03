from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import streamlit as st

#------------------------------------------
# Text Loader
loader = TextLoader("example2.txt", encoding="utf-8")
docs = loader.load()
print(docs[0].page_content)

#------------------------------------------
# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, separators="\n")
chunks = text_splitter.split_documents(docs)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} size: {len(chunk.page_content)}")
    print(chunk.page_content)

#------------------------------------------
# Embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)

#------------------------------------------
# CPU-friendly LLM
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

#------------------------------------------
# Streamlit App
st.set_page_config(page_title="Simple RAG Demo", layout="centered")
st.title("📚 Simple RAG Demo")

query = st.text_input("❓ Enter your question:")

if st.button("Submit"):
    if query.strip():
        st.success(f"👉 You asked: {query}")

        # Retrieve top 3 relevant chunks
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Format prompt and get answer
        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        answer = llm(prompt)
        st.write("💡 Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please enter a question before submitting.")
