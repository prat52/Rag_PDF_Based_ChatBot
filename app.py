import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("🦙 Llama-3.3-70B – LCEL RAG")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f: #open the file in binary mode for writing
        f.write(uploaded_file.getbuffer()) #write the raw data of file on the disk

    process_document_to_chroma_db(save_path)
    st.success("Document embedded and stored")

question = st.text_area("Ask a question from the document")

if st.button("Answer") and question:
    answer = answer_question(question)
    st.markdown("### 📄 Answer")
    st.markdown(answer)
