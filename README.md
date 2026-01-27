# VivoAssist – Manual-Aware RAG System 📘🤖

VivoAssist is a **Retrieval-Augmented Generation (RAG)** system built to answer user questions **strictly from PDF manuals** with **page-level citations**, while preventing hallucinations and cross-manual leakage.

It is designed to be **generic and production-ready**, initially tested with vehicle manuals and later intended for **telecom product manuals**.

---

## ✨ Key Features

- 📄 Ingests **multiple large PDF manuals**
- 🧩 **Hierarchical chunking** (big / mid / small)
- 🧠 Vector search using **ChromaDB (persistent)**
- 🔒 **Manual-aware retrieval** (prevents cross-manual answers)
- 🧠 Context-aware follow-up questions (sticky manual scope)
- 📑 **Page-level citations** in answers
- 🚫 Strict guard:
  If content is not in the selected manual →
  **“Not found in the manual.”**
- ⚡ Azure OpenAI powered (GPT-4o + embeddings)

---

## 🏗️ Architecture Overview

```
PDF Manuals
   ↓
PDF Loader (page-level docs)
   ↓
Hierarchical Chunking
(big / mid / small)
   ↓
ChromaDB (persistent vectors)
   ↓
LlamaIndex VectorStoreIndex
   ↓
Chat Engine (manual-aware)
```

---

## 📂 Project Structure

```
app/
 ├─ chat/
 │   └─ chat_engine.py        # Terminal chat with strict manual rules
 ├─ config/
 │   └─ settings.py           # App config + Azure OpenAI setup
 ├─ ingestion/
 │   ├─ pdf_loader.py         # Page-wise PDF loading
 |   ├─ diagram_extractor.py
 │   └─ chunker.py            # Hierarchical chunking logic
 ├─ index/
 │   ├─ chroma_store.py       # Persistent Chroma DB
 │   └─ index_builder.py      # Index build + throttling
 ├─ utils/
 │   ├─ debug.py              # Chunk + retrieval debugging
 |   ├─ manual_registry.py
 |   ├─ manual_selector.py
 |   └─ models_registry.py
 └─ main.py                   # Entry point (CLI)
data/
 └─ manuals/                  # PDF manuals
chroma_db/                    # Persistent vector store
.env                           # Azure credentials
requirements.txt
.gitignore
```

---

## 🧠 Chunking Strategy (Hierarchical)

Each PDF is split into **three levels of chunks**:

| Level | Purpose              |
| ----- | -------------------- |
| Big   | High-level context   |
| Mid   | Section-level detail |
| Small | Precise answers      |

All chunks contain metadata:

- `file_name`
- `page_number / page_label`
- `chunk_level`

This allows:

- Better recall
- Accurate citations
- Reduced hallucinations

---

## 🚀 Setup & Installation

### 1️⃣ Create virtual environment

```bash
py -V:3.10 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables

Create a `.env` file:

```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
```

⚠️ **Never commit `.env` to GitHub**

---

## ▶️ Running the App

### First run (build index)

```bash
python -m app.main
```

### Rebuild index (when manuals change)

```bash
python -m app.main --rebuild-index
```

---

## 💬 Chat Rules (Very Important)

The assistant **WILL ONLY**:

- Answer using the selected PDF manual
- Use retrieved chunks as sources
- Show page numbers when available

If information is missing:

```
Not found in the manual.
```

No guessing. No external knowledge.

## 🛠️ Debug Mode (Optional)

Enable debug in `settings.py`:

```python
debug = True
```

You’ll get:

- Chunk counts
- Sample chunk previews
- Retrieval score breakdowns

---

## 🎯 Future Improvements

- Image extraction from manuals
- Diagram-based grounding
- Better manual auto-selection
- Web UI (instead of terminal)
- Telco-scale manuals (1000+ pages)

---

