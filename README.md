# DocChat - Chat with a Document using Groq

A Retrieval-Augmented Generation (RAG) application built with **Streamlit**, featuring an interactive chat-style interface. Upload a `.txt` document through the sidebar and ask questions about its content. The application retrieves relevant information from the document and uses the **Groq API** to generate answers grounded only in the provided context.

This project presents the RAG pipeline through a conversational interface, with a sidebar for document management and retrieval settings.

## How It Works

1. **Upload** — Upload a `.txt` document through the sidebar.
2. **Chunk** — The document is split into overlapping chunks of approximately 800 characters each.
3. **Embed** — Each chunk is converted into a vector embedding locally using `sentence-transformers` (`all-MiniLM-L6-v2`). No API call is required for this step.
4. **Retrieve** — When you ask a question, it is embedded using the same model, and the most relevant chunks are retrieved using cosine similarity.
5. **Generate** — The retrieved chunks and your question are sent to the **Groq API** (`llama-3.3-70b-versatile`), which generates a grounded answer based on the retrieved context and displays it in the chat interface.

```text
.txt document → chunk → embed (local) → store in memory
                                             ↓
question (chat input) → embed (local) → similarity search → top-k chunks
                                             ↓
                              Groq LLM → grounded answer
                                             ↓
                                      chat interface

```

## Project Structure

```text
rag-streamlit-app/
├── app.py             # Main Streamlit application and chat UI
├── requirements.txt   # Python dependencies
├── .env               # Groq API key (not committed to version control)
├── .gitignore         # Files and folders excluded from Git
└── README.md          # Project documentation

```


## Prerequisites

- Python 3.9+
- A free Groq API key from [console.groq.com](https://console.groq.com)
## Screenshot

## Setup

**1. Move into the project folder:**
```bash
cd rag-streamlit-app 
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Open `.env` and replace the placeholder:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the App

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

## Usage

1. In the **sidebar**, upload a `.txt` file — it's automatically chunked and indexed.
2. Once indexed, the sidebar shows the active document name and chunk count.
3. Type a question in the **chat box** at the bottom of the main panel.
4. The assistant replies as a chat message; expand **"📚 Sources used"** under any reply to see exactly which chunks were retrieved and their similarity scores.
5. Keep chatting — the full conversation stays visible. Use **🗑️ Clear chat** in the sidebar to start over without re-uploading.
6. Uploading a new file resets the chat and re-indexes automatically.

## Configuration

Tweakable directly in `app.py`:

| Setting | Location | Default | Notes |
|---|---|---|---|
| Groq model | `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any model available in your Groq account |
| Chunk size | `chunk_text()` | 800 characters | Larger = more context per chunk, fewer chunks |
| Chunk overlap | `chunk_text()` | 150 characters | Prevents losing context at chunk boundaries |
| Chunks retrieved (top-k) | Sidebar slider | 4 | Adjustable live in the app, 1–8 |
| Embedding model | `load_embedder()` | `all-MiniLM-L6-v2` | Small and fast; swap for a larger model for better accuracy |

## Notes & Limitations

- **Single file only** — One `.txt` file can be uploaded at a time. Uploading a new file replaces the previous document and clears the chat.
- **In-memory storage** — Embeddings and chat history are stored in `st.session_state` and are not persisted to disk. Restarting the application clears the stored data.
- **First run downloads the embedding model** — The `all-MiniLM-L6-v2` model is downloaded on the first run and cached locally for subsequent use.
- **Scaling up** — For larger documents, multiple files, or persistent storage, the in-memory NumPy-based retrieval can be replaced with a vector database such as FAISS or ChromaDB.
- **Cost** — Embeddings are generated locally at no API cost. Only the final answer generation uses the Groq API.

## Troubleshooting

- **`GROQ_API_KEY not found`** — Make sure `.env` is in the same folder as `app.py` and contains a valid API key without quotes.
- **Slow first request** — The embedding model downloads on first use; subsequent runs are faster because the model is cached locally.
- **Answers say "I don't know"** — The retrieved chunks may not contain relevant information. Try increasing the **Top-K** slider or rephrasing your question.
- **Chat not clearing** — Use the **🗑️ Clear chat** button in the sidebar instead of refreshing the browser, so the indexed document remains loaded.