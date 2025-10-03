import os
import fitz
import tabula
from PIL import Image
import pytesseract
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# langchain_community.document_loaders की अब जरूरत नहीं
from langchain_text_splitters import RecursiveCharacterTextSplitter
# HuggingFaceEmbeddings का इम्पोर्ट बदला गया
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# HuggingFacePipeline का इम्पोर्ट बदला गया
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

# --- Tesseract-OCR का पाथ यहाँ सेट करें ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- UI के लिए टाइटल और हेडर ---
st.set_page_config(page_title="PDF से पूछें", layout="wide")
st.title("📄 PDF से सवाल-जवाब करें")

# --- कैशिंग का उपयोग करके परफॉरमेंस सुधार ---
# यह फंक्शन सिर्फ एक बार चलेगा और रिजल्ट को मेमोरी में रखेगा
@st.cache_resource
def create_rag_pipeline(pdf_path):
    """
    पूरी RAG पाइपलाइन बनाता है और इसे कैश में स्टोर करता है।
    """
    if not os.path.exists(pdf_path):
        st.error(f"त्रुटि: '{pdf_path}' फाइल नहीं मिली।")
        return None

    # --- 1. डेटा लोडर ---
    with st.spinner(f"'{pdf_path}' से डेटा निकाला जा रहा है..."):
        all_text = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                all_text.append(page.get_text("text"))
                pix = page.get_pixmap(dpi=200) # तेज प्रोसेसिंग के लिए DPI कम किया
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    all_text.append(ocr_text)
            
            tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            if tables:
                for table in tables:
                    all_text.append(table.to_string(index=False))
        except Exception as e:
            st.warning(f"डेटा निकालने में कुछ समस्या हुई: {e}")
        
        docs = [Document(page_content="\n".join(all_text))]

    # --- 2. टेक्स्ट चंकिंग ---
    with st.spinner("टेक्स्ट को टुकड़ों में तोड़ा जा रहा है..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

    # --- 3. एम्बेडिंग और वेक्टर DB ---
    with st.spinner("एम्बेडिंग और वेक्टर स्टोर बनाया जा रहा है..."):
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)

    # --- 4. LLM (CPU-फ्रेंडली) ---
    with st.spinner("CPU-फ्रेंडली मॉडल लोड हो रहा है..."):
        # <<--- यहाँ CPU-फ्रेंडली मॉडल सेट किया गया है ---<<
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=pipe)
    
    # --- 5. Q&A चेन ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # 3 सबसे प्रासंगिक टुकड़े ढूंढेगा
        return_source_documents=True
    )
    
    st.success("--- RAG पाइपलाइन तैयार है! ---")
    return qa_chain

# --- मुख्य UI लॉजिक ---
pdf_file_path = "attention.pdf"
rag_pipeline = create_rag_pipeline(pdf_file_path)

if rag_pipeline:
    st.header(f"अब आप '{pdf_file_path}' के बारे में सवाल पूछ सकते हैं")
    query = st.text_input("अपना सवाल यहाँ लिखें:", placeholder="What is the main idea of this document?")

    if st.button("जवाब प्राप्त करें"):
        if query.strip():
            with st.spinner("जवाब ढूंढा जा रहा है..."):
                try:
                    result = rag_pipeline({"query": query})
                    st.subheader("उत्तर:")
                    st.write(result['result'])

                    # सोर्स डॉक्यूमेंट्स दिखाने का विकल्प
                    with st.expander("उत्तर कहाँ से मिला? (सोर्स देखें)"):
                        for doc in result['source_documents']:
                            st.info(doc.page_content)
                except Exception as e:
                    st.error(f"जवाब देते समय एक त्रुटि हुई: {e}")
        else:
            st.warning("कृपया कोई सवाल दर्ज करें।")

