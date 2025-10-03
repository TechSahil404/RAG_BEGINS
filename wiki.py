from langchain_community.retrievers import WikipediaRetriever
retriver=WikipediaRetriever
docs=retriver.invoke("juice WRLD")
docs[0].page_content[:200]