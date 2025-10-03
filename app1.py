# app.py
import streamlit as st
from pathlib import Path
from rag_pipeline import load_pdf, split_documents, create_vector_db, get_llm, answer_query

st.set_page_config(page_title="PDF ↔ RAG Chat (Gemini + LangChain)", layout="wide")
st.title("📚 PDF Chat — LangChain + Gemini (Windows)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
persist_dir = "chroma_db_windows"

if uploaded_file is not None:
    file_path = Path("uploads") / uploaded_file.name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"File saved: {file_path}")

    # Backend pipeline steps
    docs = load_pdf(str(file_path))
    st.write(f"Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    st.write(f"Created {len(chunks)} chunks.")

    vectordb, embed_model = create_vector_db(chunks, persist_dir=persist_dir)
    st.success("Vector DB ready.")

    llm = get_llm()
    st.success("LLM initialized.")

    # Chat interface
    st.subheader("Ask a question about your PDF")
    question = st.text_input("Your question:")
    if st.button("Ask") and question.strip():
        answer, used_docs = answer_query(question, vectordb, llm)
        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander("Show supporting chunks"):
            for i, d in enumerate(used_docs):
                st.write(f"--- chunk {i+1} ---")
                st.write(d.page_content[:1000])
                st.write("metadata:", d.metadata)
