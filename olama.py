from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# 1. Load your text file as a document
loader = TextLoader("example.txt", encoding="utf-8")
docs = loader.load()

# 2. Initialize the Ollama LLM (assumes Ollama is running locally with your model pulled)
llm = Ollama(model="llama3.2")

# 3. Make a QA chain using the LLM
chain = load_qa_chain(llm, chain_type="stuff")  # "stuff" is the simplest chain type

# 4. Run a question against your text document
question = "What is the mobile number of sahi ?" \
"and what are all projects that sahil is working on ?" \
"in which university sahil study and which year he complete his graduation" \
"and what is the cgpa of sahil and also 12th marks "
response = chain.run(input_documents=docs, question=question)

print("Answer:", response)
