"""
Offline Streamlit RAG Resume Search Bot
---------------------------------------
- Accepts multiple resume PDFs (bulk upload)
- Extracts text, splits into chunks
- Builds or loads a FAISS vector store using HuggingFace embeddings (offline)
- Provides a search box to find specific candidates by name or keyword
- Uses a RetrievalQA chain with a local HuggingFace model to answer queries

Requirements (requirements.txt):
streamlit
langchain
langchain-community
faiss-cpu
PyPDF2
tqdm
transformers
torch
sentence-transformers

Run:
streamlit run streamlit_rag_resume_bot.py
"""

import os
import io
import pickle
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from tqdm import tqdm

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import pipeline

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast
LLM_MODEL_NAME = "distilbert-base-uncased"  # simple offline QA
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
INDEX_DIR = "resume_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index")
DOCS_PKL = os.path.join(INDEX_DIR, "docs.pkl")

# -----------------------------
# Utilities
# -----------------------------

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            metadata = {"source": filename, "page": i + 1}
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    new_docs: List[Document] = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for idx, chunk in enumerate(chunks):
            metadata = dict(d.metadata)
            metadata.update({"chunk": idx})
            new_docs.append(Document(page_content=chunk, metadata=metadata))
    return new_docs


def build_or_load_index(docs: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    os.makedirs(INDEX_DIR, exist_ok=True)
    if os.path.exists(INDEX_FILE + ".index") and os.path.exists(DOCS_PKL) and len(docs) == 0:
        try:
            st.info("Loading existing index from disk...")
            with open(DOCS_PKL, "rb") as f:
                stored_docs = pickle.load(f)
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            st.warning(f"Failed to load index: {e}. Rebuilding...")

    if not docs:
        return None

    st.info("Splitting documents into chunks...")
    split_docs = split_documents(docs)
    st.info(f"Creating embeddings for {len(split_docs)} chunks (offline)...")

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    try:
        vectorstore.save_local(INDEX_DIR)
        with open(DOCS_PKL, "wb") as f:
            pickle.dump(split_docs, f)
        st.success("Index built and saved to disk.")
    except Exception as e:
        st.warning(f"Failed to persist index: {e}")

    return vectorstore


# -----------------------------
# Candidate matching helpers
# -----------------------------

def simple_name_sniff(text: str) -> str:
    lines = text.splitlines()
    for line in lines[:10]:
        line_clean = line.strip()
        if not line_clean:
            continue
        lower = line_clean.lower()
        if lower.startswith("name:") or lower.startswith("candidate:"):
            return line_clean.split(":", 1)[1].strip()
    for line in lines[:6]:
        line_clean = line.strip()
        if 2 <= len(line_clean.split()) <= 4 and any(c.isalpha() for c in line_clean):
            return line_clean
    return ""


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Offline RAG Resume Search Bot", layout="wide")
    st.title("🧾 Offline RAG Resume Search Bot — LangChain + Streamlit")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "docs_count" not in st.session_state:
        st.session_state.docs_count = 0

    uploaded_files = st.file_uploader("Upload resume PDFs (multiple allowed)", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Build / Rebuild Index from Uploads"):
            if not uploaded_files:
                st.error("Please upload 1 or more PDF files first.")
            else:
                all_docs: List[Document] = []
                progress_bar = st.progress(0)
                for idx, f in enumerate(uploaded_files):
                    try:
                        bytes_read = f.read()
                        docs = extract_text_from_pdf(bytes_read, f.name)
                        all_docs.extend(docs)
                    except Exception as e:
                        st.warning(f"Failed to read {f.name}: {e}")
                    progress_bar.progress(int((idx + 1) / len(uploaded_files) * 100))

                if all_docs:
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                    vs = build_or_load_index(all_docs, embeddings)
                    st.session_state.vectorstore = vs
                    st.session_state.docs_count = len(all_docs)
                else:
                    st.error("No text extracted from uploaded PDFs.")

        if st.button("Load Existing Index (if present)"):
            try:
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                vs = build_or_load_index([], embeddings)
                if vs is None:
                    st.error("No existing index found. Please build an index first.")
                else:
                    st.session_state.vectorstore = vs
                    st.session_state.docs_count = "(loaded)"
            except Exception as e:
                st.error(f"Error loading index: {e}")

        st.markdown("---")
        st.write("**Index status:**")
        st.write(f"Built documents: {st.session_state.docs_count}")

    with col2:
        st.header("Search / Ask")
        query = st.text_input("Search candidate by name or ask about skills/experience")
        top_k = st.slider("Number of results", 1, 10, 4)

        if st.button("Search"):
            if not query:
                st.error("Please enter a query.")
            elif st.session_state.vectorstore is None:
                st.error("No index available. Upload PDFs and build the index first.")
            else:
                with st.spinner("Searching..."):
                    docs = st.session_state.vectorstore.similarity_search(query, k=top_k)

                    if not docs:
                        st.info("No matches found.")
                    else:
                        st.subheader("Top matching chunks")
                        for i, d in enumerate(docs, start=1):
                            name_guess = simple_name_sniff(d.page_content)
                            st.markdown(f"**Result {i} — Source:** {d.metadata.get('source')} (page {d.metadata.get('page')})")
                            if name_guess:
                                st.markdown(f"- **Name (sniffed):** {name_guess}")
                            st.markdown(f"- **Snippet:** {d.page_content[:800]}\n---")

                        # Offline QA using HuggingFace pipeline
                        qa_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=200)
                        llm = HuggingFacePipeline(pipeline=qa_pipeline)
                        qa = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                        )
                        try:
                            answer = qa.run(query)
                            st.subheader("🔎 Bot Answer")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Failed to run QA chain: {e}")

    st.markdown("---")
    st.caption("Runs fully offline using HuggingFace embeddings + local transformer model.")


if __name__ == "__main__":
    main()
