# -*- coding: utf-8 -*-
"""MedPolRAG.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e5WrCF2eSeI8Q1MI8y6ZdVbc2JjPct5W
"""
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key from .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📄 Health Policy Q&A (RAG App)")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a health policy PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and chunk document
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    if not documents:
        st.warning("⚠️ No content extracted from the uploaded PDF. Please try a different file.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Ask a question
    quest = st.text_input("Ask a question about the policy")

    if quest:
        result = qa_chain(quest)

        # Display answer
        st.markdown(f"### 🧠 Answer:\n{result['result']}")

        # Display source chunks
        with st.expander("📄 Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source Chunk {i+1}** (Page {doc.metadata.get('page', 'unknown')}):")
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
