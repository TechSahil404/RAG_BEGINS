from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader(video_id="dQw4w9WgXcQ", add_video_info=False)
docs = loader.load()

print(len(docs))
print(docs[0].page_content[:500])
print(docs[0].metadata)