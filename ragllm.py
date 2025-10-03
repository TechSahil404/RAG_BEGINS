

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


with open("example2.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # smaller chunk for memory
    chunk_overlap=50
)
chunks = text_splitter.split_text(raw_text)


embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embedding_model)


model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    device=-1  # CPU
)

llm = HuggingFacePipeline(pipeline=pipe)


query = "What is the cgpa of sahil kumawat?"
retrieved_docs = vector_store.similarity_search(query, k=1)  # top 1 chunk for CPU

context = "\n".join([doc.page_content for doc in retrieved_docs])
prompt = f"Answer the question based on context:\nContext:\n{context}\nQuestion: {query}\nAnswer in one sentence:"

# Use invoke() instead of deprecated __call__()
answer = llm.invoke(prompt)
print("Answer:", answer)
