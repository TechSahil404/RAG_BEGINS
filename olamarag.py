import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#--------------------------------------------------------------------------------------------------------------
# --- 1. SETUP OLLAMA MODELS ---
# Initialize the LLM for generation
llm = Ollama(model="llama3.2")
# Initialize the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. LOAD AND PROCESS THE DOCUMENT ---
# Load your local text file
loader = TextLoader("example.txt", encoding="utf-8")
docs = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# --- 3. CREATE AND STORE EMBEDDINGS (VECTOR STORE) ---
# Create a FAISS vector store from the document chunks and embeddings
# This will create and save a local vector database
vector_store = FAISS.from_documents(split_documents, embeddings)

# --- 4. DEFINE THE RAG CHAIN ---
# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Define the prompt template for the RAG chain
prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based only on the following context. 
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question: {input}
    """
)

# Create the main document chain that will combine the retrieved context and the question
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the final retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- 5. ASK A QUESTION ---
question = "What is the cgpa of sahil kumawat?" \
"mobile number of sahil kumawat" \
"university name of sahil kumawat where he studied " \
"name of projects that sahil kumawat is working on" \
"name of all skills of sahil kumawat " \
"and in which skill sahil have much intrest "
response = retrieval_chain.invoke({"input": question})

print("### Question:")
print(question)
print("\n### Answer:")
print(response["answer"])
