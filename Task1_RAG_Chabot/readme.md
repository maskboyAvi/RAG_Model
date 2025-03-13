# TV Industry Report RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that extracts data from a TV Industry Report PDF and answers queries about various metrics (e.g., network speeds, anomalies, trends) for any specified year. It uses FAISS for context retrieval and leverages Ollama’s Llama3.2 model for text generation via the CLI.

## Prerequisites

Before setting up this project, ensure you have the following installed on your local computer:
- **Ollama CLI**: Install Ollama from [ollama.com](https://ollama.com) and make sure you have the **llama3.2** model available.
  - *Note*: Ollama and its models are not Python packages and should be installed via their own CLI installer. Do not add them to your `requirements.txt`.
- **Python 3.8 or higher** (for creating the virtual environment).

## Quick Setup

If you just want to run the chatbot with the provided FAISS index and chunks file, simply run:
```bash
streamlit run app.py
```

## Training on Your Own PDF

If you want to generate your own FAISS index and text chunks from a PDF (for example, if you want to use a different PDF), run:
```bash
python model.py
```
This script will:
- Extract text from your PDF (make sure your PDF file is named `TV_Industry_Report.pdf` and placed in the project directory).
- Split the text into overlapping chunks.
- Compute embeddings using SentenceTransformer.
- Build and save a FAISS index (`faiss_index.index`) and the corresponding text chunks (`chunks.pkl`).

## Project Setup Details

1. **Clone the repository and navigate into the project directory.**
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate   # On Windows, use: rag_env\Scripts\activate
   ```
3. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure Ollama CLI is installed and configured with llama3.2.**  
   Visit [ollama.com](https://ollama.com) to download and set up Ollama on your computer.

5. **Run the FAISS index script (if training on your own PDF):**
   ```bash
   python model.py
   ```
6. **Launch the chatbot UI:**
   ```bash
   streamlit run app.py
   ```

## Additional Notes

- **Ollama CLI and Model Installation:**  
  Ollama and the llama3.2 model are installed via their own installers (or using CLI commands provided by Ollama). These are not Python packages and should not be added to `requirements.txt`. Please follow the official instructions on [ollama.com](https://ollama.com) to set these up.

- **Project Structure:**  
  - `app.py`: The main Streamlit application with a chat UI.  
  - `model.py`: The script to process your PDF and create the FAISS index and text chunks.  
  - `chunks.pkl` & `faiss_index.index`: Pre-generated files (if provided) for quick testing.

- **Customization:**  
  The chatbot dynamically extracts the year from your query and constructs prompts accordingly. It can answer questions regarding any year based on the context extracted from the PDF.
