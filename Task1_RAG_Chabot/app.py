# # app.py
# # ------------
# # This script implements a UI-based RAG chatbot using Streamlit.
# # It loads the FAISS index and text chunks, retrieves relevant context from the TV Industry Report,
# # builds a prompt for the question, and calls Ollama’s Llama 3.2 model via a subprocess call.
# # The chat UI is styled with black text on a white background, with user messages on the right and bot messages on the left.

# import os
# import nest_asyncio
# nest_asyncio.apply()  # Patch the event loop for compatibility with Streamlit

# # Suppress Torch warnings (if any)
# import torch
# torch.classes.__path__ = []

# import pickle
# import numpy as np
# import faiss
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import subprocess

# def query_ollama(prompt: str) -> str:
#     """
#     Calls the Ollama CLI to generate a response using the Llama 3.2 model.
#     Adjust the command if needed.
#     """
#     try:
#         # Use "ollama run" instead of "ollama chat"
#         result = subprocess.run(
#             ["ollama", "run", "llama3.2"],
#             input=prompt,
#             text=True,
#             encoding="utf-8",  # Ensure proper Unicode encoding
#             capture_output=True,
#             check=True  # Raise exception if the command fails
#         )
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         return f"Error in Ollama CLI call: {e.stderr.strip()}"


# def load_resources():
#     """Load FAISS index, text chunks, and the embedding model."""
#     # Load FAISS index and text chunks from saved files
#     faiss_index = faiss.read_index("faiss_index.index")
#     with open("chunks.pkl", "rb") as f:
#         text_chunks = pickle.load(f)
#     # Load embedding model for encoding queries
#     embed_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return faiss_index, text_chunks, embed_model

# def retrieve_context(query, faiss_index, text_chunks, embed_model, top_k=7):
#     """Retrieve top_k relevant text chunks for the given query."""
#     query_embedding = embed_model.encode([query], convert_to_numpy=True)
#     distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
#     retrieved_chunks = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]
#     return retrieved_chunks

# def generate_answer(prompt: str) -> str:
#     """Generate an answer using Ollama's Llama 3.2 via our helper function."""
#     response = query_ollama(prompt)
#     return response

# def display_chat():
#     """Display the chat conversation with custom CSS styling."""
#     for chat in st.session_state.chat_history:
#         if chat["role"] == "bot":
#             st.markdown(f'<div class="chat-bubble bot-bubble">{chat["message"]}</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="chat-bubble user-bubble">{chat["message"]}</div>', unsafe_allow_html=True)

# def main():
#     st.set_page_config(page_title="TV Industry Report Chatbot", layout="wide")
#     st.markdown("""
#     <style>
#     .chat-bubble {
#         padding: 10px;
#         margin: 10px;
#         border-radius: 10px;
#         max-width: 70%;
#         color: black;
#         font-size: 16px;
#     }
#     .user-bubble {
#         background-color: #DCF8C6;
#         margin-left: auto;
#         text-align: right;
#     }
#     .bot-bubble {
#         background-color: #F1F0F0;
#         margin-right: auto;
#         text-align: left;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.title("TV Industry Report Chatbot")
    
#     # Initialize chat history if not already set
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
    
#     # Load resources only once
#     if "resources_loaded" not in st.session_state:
#         (st.session_state.faiss_index,
#          st.session_state.text_chunks,
#          st.session_state.embed_model) = load_resources()
#         st.session_state.resources_loaded = True
    
#     # Chat input form
#     with st.form(key="chat_form", clear_on_submit=True):
#         user_input = st.text_input("Enter your query:")
#         submitted = st.form_submit_button("Send")
    
#     if submitted and user_input:
#         # Append the user message
#         st.session_state.chat_history.append({"role": "user", "message": user_input})
        
#         # Retrieve context from the FAISS index
#         context_chunks = retrieve_context(user_input, st.session_state.faiss_index,
#                                           st.session_state.text_chunks, st.session_state.embed_model)
#         combined_context = " ".join(context_chunks)
        
#         # Build a prompt instructing the model to extract exact data
#         prompt = (
#             "Based solely on the context provided from the TV Industry Report, extract and format the precise data for network speeds in 2018. "
#             "The answer should exactly follow this format (if available):\n"
#             "'According to the report, the average fixed broadband download speed in 2018 was <download_speed> Mbit/s, and superfast broadband "
#             "connections reached <superfast_connections> million, accounting for <percentage>% of total fixed broadband lines.'\n\n"
#             "Context: " + combined_context + "\n"
#             "Question: " + user_input
#         )
        
#         # Generate answer using Ollama's Llama 3.2 model
#         answer = generate_answer(prompt)
        
#         # Append the bot response
#         st.session_state.chat_history.append({"role": "bot", "message": answer})
    
#     display_chat()

# if __name__ == "__main__":
#     main()


# app.py
# ------------
# This script implements a UI-based RAG chatbot using Streamlit.
# It loads the FAISS index and text chunks (which include plain text, table data, etc.),
# retrieves relevant context from the TV Industry Report,
# dynamically extracts the year from the user query,
# builds a prompt based on the query type (data extraction or anomaly/trend analysis),
# and calls Ollama’s Llama 3.2 model via the CLI.
# The UI is styled with black text on a white background, with user messages on the right and bot messages on the left.

import os
import re
import nest_asyncio
nest_asyncio.apply()  # Patch the event loop for compatibility with Streamlit

# Suppress Torch warnings (if any)
import torch
torch.classes.__path__ = []

import pickle
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import subprocess

def query_ollama(prompt: str) -> str:
    """
    Calls the Ollama CLI to generate a response using the Llama 3.2 model.
    Uses "ollama run llama3.2" command; ensure the Ollama CLI is installed and configured.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=prompt,
            text=True,
            encoding="utf-8",  # Ensure proper Unicode encoding
            capture_output=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error in Ollama CLI call: {e.stderr.strip()}"

def load_resources():
    """Load FAISS index, text chunks, and the embedding model."""
    faiss_index = faiss.read_index("faiss_index.index")
    with open("chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return faiss_index, text_chunks, embed_model

def retrieve_context(query, faiss_index, text_chunks, embed_model, top_k=7):
    """Retrieve top_k relevant text chunks for the given query."""
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
    retrieved_chunks = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]
    return retrieved_chunks

def generate_answer(prompt: str) -> str:
    """Generate an answer using Ollama's Llama 3.2 via our helper function."""
    response = query_ollama(prompt)
    return response

def display_chat():
    """Display the chat conversation with custom CSS styling."""
    for chat in st.session_state.chat_history:
        if chat["role"] == "bot":
            st.markdown(f'<div class="chat-bubble bot-bubble">{chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble user-bubble">{chat["message"]}</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="TV Industry Report Chatbot", layout="wide")
    st.markdown("""
    <style>
    .chat-bubble {
        padding: 10px;
        margin: 10px;
        border-radius: 10px;
        max-width: 70%;
        color: black;
        font-size: 16px;
    }
    .user-bubble {
        background-color: #DCF8C6;
        margin-left: auto;
        text-align: right;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        margin-right: auto;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("TV Industry Report Chatbot")
    
    # Initialize chat history if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Load resources only once
    if "resources_loaded" not in st.session_state:
        (st.session_state.faiss_index,
         st.session_state.text_chunks,
         st.session_state.embed_model) = load_resources()
        st.session_state.resources_loaded = True
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your query:")
        submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        # Append the user's message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        
        # Extract a year from the query (e.g., 2010 to 2023)
        year_match = re.search(r'\b(20\d{2})\b', user_input)
        query_year = year_match.group(1) if year_match else None
        
        # Retrieve context from the FAISS index
        context_chunks = retrieve_context(user_input, st.session_state.faiss_index,
                                          st.session_state.text_chunks, st.session_state.embed_model)
        combined_context = " ".join(context_chunks)
        
        # Build a dynamic prompt based on the query
        if query_year:
            # Check if query asks for trends or anomalies
            if any(kw in user_input.lower() for kw in ["anomaly", "anomalies", "trend", "trends", "noteworthy"]):
                prompt = (
                    f"Based solely on the context provided from the TV Industry Report (which includes both plain text and tabular data from pages covering years 2010 to 2023), "
                    f"analyze the data for the year {query_year}. Identify any anomalies in {query_year} compared to previous and subsequent years and highlight noteworthy trends. "
                    f"Focus on key metrics such as subscriber counts, network speeds, data usage, revenue, and call volumes. Provide a concise, data-backed answer with specific numerical values where available.\n\n"
                    "Context: " + combined_context + "\n"
                    "Question: " + user_input
                )
            else:
                # Default: extract specific data for the queried year
                prompt = (
                    f"Based solely on the context provided from the TV Industry Report, extract and format the precise data for network speeds in {query_year}. "
                    "The answer should exactly follow this format (if available):\n"
                    f"'According to the report, the average fixed broadband download speed in {query_year} was <download_speed> Mbit/s, and superfast broadband "
                    f"connections reached <superfast_connections> million, accounting for <percentage>% of total fixed broadband lines.'\n\n"
                    "Context: " + combined_context + "\n"
                    "Question: " + user_input
                )
        else:
            # If no year is mentioned, use a generic prompt
            prompt = (
                "Based solely on the context provided from the TV Industry Report, extract and format the precise data relevant to the question. "
                "Context: " + combined_context + "\n"
                "Question: " + user_input
            )
        
        # Generate answer using Ollama's Llama 3.2 model
        answer = generate_answer(prompt)
        
        # Append the bot's response to chat history
        st.session_state.chat_history.append({"role": "bot", "message": answer})
    
    display_chat()

if __name__ == "__main__":
    main()
