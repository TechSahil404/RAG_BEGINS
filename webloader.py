from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.espn.com/")
docs = loader.load()

print(docs)
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

from langchain_community.document_loaders import PyPDFLoader 
loader=PyPDFLoader("agentic.pdf")
docs=loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)