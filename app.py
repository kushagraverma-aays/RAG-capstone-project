import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Setup
load_dotenv()  # loads GROQ_API_KEY from .env

st.set_page_config(page_title="DocChat · RAG with Groq", page_icon="💬", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + capable Groq-hosted model

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# Cached embedding model loader
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


# Helper functions
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)


def retrieve_top_k(query: str, chunks, chunk_embeddings, k: int = 4):
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = chunk_embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def ask_groq(question: str, context_chunks):
    context_text = "\n\n---\n\n".join([c for c, _ in context_chunks])

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided "
        "context from a document. If the answer is not contained in the context, "
        "say you don't know based on the document. Be concise and accurate."
    )

    user_prompt = f"""Context from the document:
{context_text}

Question: {question}

Answer the question using only the context above."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# Session state
defaults = {
    "chunks": None,
    "embeddings": None,
    "file_name": None,
    "messages": [],  # chat-style history: list of {role, content, sources?}
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# Sidebar — document upload & settings

with st.sidebar:
    st.title("💬 DocChat")
    st.caption("Chat with your document, powered by Groq")

    st.divider()

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

        with st.spinner("Reading and indexing document..."):
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []

        st.success(f"Indexed '{uploaded_file.name}' ({len(chunks)} chunks)")

    if st.session_state.file_name:
        st.markdown(f"**Active document:** `{st.session_state.file_name}`")
        st.markdown(f"**Chunks indexed:** {len(st.session_state.chunks)}")

    st.divider()

    top_k = st.slider("Chunks retrieved per question", min_value=1, max_value=8, value=4)
    show_sources = st.checkbox("Show retrieved context with answers", value=True)

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat interface
st.header("Chat with your document")

if st.session_state.chunks is None:
    st.info("👈 Upload a `.txt` file in the sidebar to start chatting.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("📚 Sources used"):
                    for i, (chunk, score) in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Chunk {i}** · similarity `{score:.3f}`")
                        st.text(chunk)

    # Chat input
    user_question = st.chat_input("Ask something about the document...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                top_chunks = retrieve_top_k(
                    user_question, st.session_state.chunks, st.session_state.embeddings, k=top_k
                )
                answer = ask_groq(user_question, top_chunks)

            st.markdown(answer)
            if show_sources:
                with st.expander("📚 Sources used"):
                    for i, (chunk, score) in enumerate(top_chunks, start=1):
                        st.markdown(f"**Chunk {i}** · similarity `{score:.3f}`")
                        st.text(chunk)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": top_chunks}
        )
