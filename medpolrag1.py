import os
from getpass import getpass
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Setup API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📄 Health Policy Q&A (RAG App)")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a health policy PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split document
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Setup retriever and LLM
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # User query
    query = st.text_input("Ask a question about the policy")
    if query:
        result = qa_chain(query)
        st.markdown(f"### 🧠 Answer\n{result['result']}")

        with st.expander("📄 Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'unknown')}):")
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
