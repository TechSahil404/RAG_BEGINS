from langchain_community.document_loaders import TextLoader 
from langchain_community.document_loaders import PyPDFLoader

loader=TextLoader("example2.txt",encoding="utf-8")
docs=loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)

loader=PyPDFLoader("agentic.pdf")
pdf=loader.load()
print(pdf)
print(len(pdf))
print(pdf[0].page_content)

