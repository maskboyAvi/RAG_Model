# model.py
# ------------
# This script extracts text from the TV Industry Report PDF,
# splits it into overlapping chunks (for precise retrieval),
# computes embeddings using SentenceTransformer,
# builds a FAISS index, and saves both the index and text chunks.

import os
import re
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def load_pdf(file_path):
    """Extract text from each page of the PDF."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "
    return full_text

def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split text into chunks of approximately 'chunk_size' words,
    with an overlap of 'overlap' words to preserve context.
    """
    words = re.findall(r'\S+', text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def build_faiss_index(chunks, embedding_model):
    """Compute embeddings for each chunk and build a FAISS index."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

if __name__ == "__main__":
    pdf_file = "TV_Industry_Report.pdf"  # Place the PDF in the same folder
    print("Loading PDF...")
    pdf_text = load_pdf(pdf_file)
    
    print("Chunking text...")
    text_chunks = chunk_text(pdf_text)
    
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Building FAISS index...")
    index, _ = build_faiss_index(text_chunks, embed_model)
    
    # Save the index and text chunks for later retrieval
    faiss.write_index(index, "faiss_index.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)
    
    print("Model setup complete and index saved.")
