from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="country.csv", encoding="utf-8")
docs=loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)