Offline RAG Resume Search Bot — LangChain + Streamlit

A fully offline Resume Search Bot built with LangChain, Streamlit, and FAISS, designed for bulk resume screening without internet dependency.

🚀 Features

📂 Bulk Upload Resumes – Accepts multiple PDF files at once

🔍 Text Extraction & Chunking – Extracts text from resumes and splits into searchable chunks

📚 Offline FAISS Vector Store – Builds or loads FAISS index with HuggingFace embeddings

🧑‍💻 Smart Candidate Search – Search by name, skills, or keywords

🤖 Local RetrievalQA Chain – Uses a local HuggingFace model (no API needed) to answer queries

🛠️ Tech Stack & Packages

Streamlit
 – Interactive UI

LangChain
 – RAG pipeline and chain management

LangChain-Community
 – Community modules for embeddings & FAISS

FAISS
 – Vector store for similarity search

PyPDF2
 – PDF text extraction

Transformers
 – Local HuggingFace models

Torch
 – Backend for HuggingFace models

Sentence-Transformers
 – Embeddings for resume chunks

tqdm
 – Progress bar for file processing
