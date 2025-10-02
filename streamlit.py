import os
import json
import glob
import shutil
import faiss
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
from playwright.sync_api import sync_playwright
import trafilatura
import subprocess

# Make sure Playwright browsers are installed
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    subprocess.run(["pip", "install", "playwright"], check=True)
    subprocess.run(["playwright", "install", "chromium"], check=True)
    from playwright.sync_api import sync_playwright

# ================================
# CONFIG
# ================================
RESULTS_DIR = "results"
CHUNKS_DIR = "chunks"
INDEX_DIR = "faiss_index"


# ================================
# CLEANUP FUNCTION
# ================================
def clear_old_data():
    """Delete old results, chunks, and FAISS index before new website run."""
    for d in [RESULTS_DIR, CHUNKS_DIR, INDEX_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


# ================================
# PLAYWRIGHT FETCHER
# ================================
def fetch_page_with_playwright(url, timeout=15000):
    """Fetch fully rendered HTML using Playwright headless browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(timeout)
        page.goto(url)
        content = page.content()
        browser.close()
    return content


# ================================
# HELPERS
# ================================
def get_resource_type(url):
    if url.endswith('.pdf'): return 'pdf'
    if url.endswith('.docx'): return 'docx'
    if url.endswith('.doc'): return 'doc'
    return 'html'

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and \
           parsed.scheme in ['http', 'https'] and not parsed.fragment


# ================================
# STEP 1: Crawl Website
# ================================
def crawl_website(start_url, max_depth):
    urls_to_visit = [(start_url, 0)]
    visited_urls = set()
    crawled_resources = []

    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)

        if current_url in visited_urls or current_depth > max_depth:
            continue

        try:
            visited_urls.add(current_url)
            html_content = fetch_page_with_playwright(current_url)

            resource_info = {
                'source_url': current_url,
                'type': get_resource_type(current_url),
                'depth': current_depth,
            }
            crawled_resources.append(resource_info)

            if resource_info['type'] == 'html' and current_depth < max_depth:
                soup = BeautifulSoup(html_content, 'html.parser')
                for link_tag in soup.find_all('a', href=True):
                    href = link_tag['href']
                    absolute_url = urljoin(current_url, href)
                    if is_valid_url(absolute_url):
                        if urlparse(absolute_url).netloc == urlparse(start_url).netloc:
                            if absolute_url not in visited_urls:
                                urls_to_visit.append((absolute_url, current_depth + 1))

        except Exception as e:
            st.warning(f"Error crawling {current_url}: {e}")

    return crawled_resources


# ================================
# STEP 2: Extract Text (Trafilatura)
# ================================
def process_with_trafilatura(resources, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    for i, res in enumerate(resources, 1):
        if res['type'] != 'html':
            continue
        url = res['source_url']
        try:
            html_content = fetch_page_with_playwright(url)
            extracted_text = trafilatura.extract(html_content, include_tables=True)
            if not extracted_text:
                continue
            output_file = os.path.join(results_dir, f"page_{i}-output.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)
        except Exception as e:
            st.warning(f"Error processing {url}: {e}")


# ================================
# STEP 3: Load & Chunk Documents
# ================================
def load_documents(results_dir=RESULTS_DIR):
    docs = []
    for file in os.listdir(results_dir):
        if file.endswith("-output.txt"):
            with open(os.path.join(results_dir, file), "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    docs.append({"text": text.strip(), "source": file})
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({"text": chunk, "source": d["source"]})
    return chunks

def save_chunks(chunks, chunks_dir=CHUNKS_DIR):
    os.makedirs(chunks_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(chunks_dir, f"chunks_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    return output_file


# ================================
# STEP 4: Embedding + FAISS Index
# ================================
def load_latest_chunks(chunks_dir=CHUNKS_DIR):
    files = glob.glob(os.path.join(chunks_dir, "chunks_*.json"))
    if not files:
        raise FileNotFoundError("No chunk files found.")
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks, latest_file

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", device="cpu"):
    model = SentenceTransformer(model_name, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_faiss_index(embeddings, chunks, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "faiss_index.bin"))
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)


# ================================
# STEP 5: Retrieval + Gemini
# ================================
def load_faiss_index(model_name="all-MiniLM-L6-v2", device="cpu"):
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
    with open(os.path.join(INDEX_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(model_name, device=device)
    return index, metadata, model

def retrieve(query, top_k=3):
    index, metadata, model = load_faiss_index()
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [metadata[idx]["text"] for idx in indices[0]]

def ask_gemini(query, api_key):
    if not api_key:
        raise ValueError("❌ No API key provided. Set it in secrets or sidebar.")
    client = genai.Client(api_key=api_key)
    context_chunks = retrieve(query, top_k=3)
    context = "\n".join(context_chunks)
    prompt = f"""
    You are a helpful assistant. Use the following context to answer:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="Website QA Bot", layout="wide")
st.title("🌐 Website QA Bot (Trafilatura + FAISS + Gemini)")

# 🔑 API Key Handling
api_key = None
if "GEMINI_API_KEY" in st.secrets:   # ✅ Streamlit Cloud
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")  # ✅ Local dev
    if not api_key:
        api_key = st.sidebar.text_input("Enter GEMINI_API_KEY:", type="password")  # ✅ Fallback


# Crawl form
with st.form("crawl_form"):
    url = st.text_input("Enter website URL:")
    depth = st.number_input("Max crawl depth", min_value=0, max_value=3, value=0)
    submitted = st.form_submit_button("Process Website")

if submitted and url:
    clear_old_data()
    st.info("Crawling website... please wait ⏳")
    resources = crawl_website(url, depth)
    st.success(f"Crawled {len(resources)} pages")

    st.info("Extracting text...")
    process_with_trafilatura(resources)

    st.info("Chunking text...")
    docs = load_documents()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    st.success(f"Created {len(chunks)} chunks")

    st.info("Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if str(model.device) == "cuda" else "cpu"
    embeddings = embed_chunks(chunks, device=device)
    save_faiss_index(embeddings, chunks)
    st.success("FAISS index ready ✅")

# Chat UI
if os.path.exists(os.path.join(INDEX_DIR, "faiss_index.bin")):
    st.subheader("💬 Ask Questions")
    query = st.text_input("Your question:")
    if st.button("Ask") and query:
        try:
            answer = ask_gemini(query, api_key)
            st.markdown(f"**🤖 Answer:** {answer}")
        except Exception as e:
            st.error(f"Error: {e}")

