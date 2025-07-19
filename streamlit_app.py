import streamlit as st
import os
import fitz  # PyMuPDF
import tiktoken
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import ollama
import torch
print(f"Torch cuda available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.nn.Linear(10, 5)
model = model.to("cuda")  # If this fails, it’s a CUDA/driver issue

# --- Config ---
SAVE_DIR = "uploaded_files"
EMBED_DIM = 384  # for sentence-transformers 'all-MiniLM-L6-v2'
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
RERANKER_MODEL  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
MODEL = "gemma:2b"  # Change to your desired model (e.g. 'mistral', 'gemma')
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Functions ---
def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokenizer.decode(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    return EMBED_MODEL.encode(chunks)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(embeddings))
    return index

def get_top_chunks2(query, chunks, index, k=3):
    q_emb = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

def get_top_chunks(query, chunks, index, initial_k=10, final_k=3):
    """
    Retrieve top-N with FAISS, rerank with cross-encoder, return top-K.
    """
    # Step 1: FAISS retrieval
    q_emb = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(q_emb), initial_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Step 2: Rerank using cross-encoder
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = RERANKER_MODEL.predict(pairs)

    # Step 3: Sort by score
    reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in reranked[:final_k]]

    return top_chunks

def ask_ollama(context, query, model=MODEL):
    full_prompt = (
        f"Answer the following question using the provided context.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}"
    )
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": full_prompt}
    ])
    return response['message']['content']

# --- UI ---
st.set_page_config(page_title="🧠 Local RAG Chat", layout="wide")
st.sidebar.title("📁 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Save PDFs
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(SAVE_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success(f"{len(uploaded_files)} file(s) saved")

# Show saved files
st.sidebar.subheader("📂 Saved Files")
pdf_files = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(".pdf")]
for fname in pdf_files:
    st.sidebar.write(f"📄 {fname}")

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load + process PDFs
if st.button("📚 Build Knowledge Base"):
    if not pdf_files:
        st.warning("No PDFs found.")
    else:
        all_chunks = []
        for fname in pdf_files:
            text = extract_pdf_text(os.path.join(SAVE_DIR, fname))
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        with st.spinner("🔢 Embedding chunks..."):
            embeddings = embed_chunks(all_chunks)
            faiss_index = create_faiss_index(embeddings)
            st.success(f"✅ Embedded {len(all_chunks)} chunks.")

        st.session_state["chunks"] = all_chunks
        st.session_state["index"] = faiss_index

# Chat interface
st.title("💬 Ask Your Documents")

if "chunks" in st.session_state and "index" in st.session_state:
    query = st.text_input("Ask a question about your PDFs:")
    if query:
        top_chunks = get_top_chunks(query, st.session_state["chunks"], st.session_state["index"])
        context = "\n\n".join(top_chunks)
        # st.markdown("#### Retrieved Context:")
        # st.info(context[:1500] + ("..." if len(context) > 1500 else ""))
        with st.spinner("🧠 Thinking..."):
            answer = ask_ollama(context, query)
        st.markdown("#### 🤖 Answer:")
        st.success(answer)
else:
    st.info("Upload and embed PDFs to start asking questions.")
