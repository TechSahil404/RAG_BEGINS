from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

loader=UnstructuredPDFLoader("agentic.pdf" , mode="elements",strategy="hi_res")
doc=loader.load()
print(doc)
print(len(doc))
print(doc[0].page_content)


from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    "agentic.pdf",
    mode="elements",
    strategy="hi_res",           # render + OCR
    infer_table_structure=True,  # table HTML/structure
    extract_images=True          # include image elements
)

docs = loader.load()
print("elements:", len(docs))
print("first.meta:", docs.metadata)
print("first.text:", docs.page_content[:300])





