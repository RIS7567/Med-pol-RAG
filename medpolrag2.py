import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📚 Multi-Document Q&A (RAG App)")

# Load all PDFs from the "docs" directory
docs_dir = "docs"
documents = []

if os.path.exists(docs_dir):
    for file in os.listdir(docs_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_dir, file))
            docs = loader.load()
            documents.extend(docs)
else:
    st.error("📁 'docs/' folder not found. Please create it and add PDFs.")

if documents:
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Ask a question
    quest = st.text_input("Ask a question across all documents")

    if quest:
        result = qa_chain(quest)

        st.markdown(f"### 🧠 Answer:\n{result['result']}")

        with st.expander("📄 Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', 'unknown')}):")
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
else:
    st.info("Upload your PDFs to the 'docs/' folder to begin.")
