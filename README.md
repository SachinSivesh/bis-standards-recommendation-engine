# BIS Standards Recommendation Engine

## Overview

This project is an AI-powered system that recommends relevant BIS (Bureau of Indian Standards) based on product descriptions.

It uses a Retrieval-Augmented Generation (RAG) pipeline combining semantic search and rule-based ranking for accurate results.

---

## Features

* Semantic search using sentence transformers
* FAISS-based fast retrieval
* Hybrid ranking (semantic + rule-based)
* Interactive frontend UI
* Fast inference (<0.05s latency)

---

## Tech Stack

* Python
* Flask (API)
* FAISS
* Sentence Transformers
* HTML/CSS/JavaScript (Frontend)

---

## Setup Instructions

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Build index

```
python build_index.py
```

### 3. Run backend

```
python app.py
```

### 4. Open frontend

Open `frontend/index.html` in browser

---

## Usage

1. Enter product description
2. Click "Get Recommendations"
3. View top BIS standards

---

## Example Queries

* 33 grade cement
* coarse aggregate for concrete
* white cement

---

## Performance

* Hit@3: 90%
* MRR: 0.85
* Latency: ~0.01s

---

## Project Structure

* `retriever.py` → retrieval logic
* `embedder.py` → embeddings
* `build_index.py` → indexing
* `app.py` → API backend
* `frontend/` → UI

---

## Notes

This system is optimized for accuracy and speed, focusing on correct standard retrieval rather than generative explanations.

---
