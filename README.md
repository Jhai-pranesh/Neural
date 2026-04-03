# NeuralSearch Prime

This project is a fully functional, two-stage e-commerce search system that uses an external retrieved catalog and locally reranks the results via semantic similarity (using `sentence-transformers`).

## Architecture Overview

**User Query** → **Streamlit Frontend** → **FastAPI Backend** → **SerpApi (Amazon Engine Retrieval)** → **Local Reranking (Semantic + Keyword)** → **Card UI Display**

## Features

- **Hybrid Retrieval**: Queries standard external APIs first.
- **Neural Reranking**: Locally reranks the response locally using `all-MiniLM-L6-v2` (`0.7 * semantic_similarity + 0.3 * keyword_overlap`).
- **Rank Comparison**: Shows explicit UI arrows (↑/↓) denoting rank improvements or demotions.
- **Modern UI**: Provided by Streamlit featuring card layout and interactive tabs.

## Setup Instructions

1. **Install Requirements**
   Ensure Python 3.10+ is available.
   ```bash
   pip install -r requirements.txt
   ```

2. **Supply the API Key**
   Add your SerpAPI Key to the root `.env` file:
   ```env
   SERPAPI_KEY=YOUR_API_KEY
   ```

3. **Start the Backend Server**
   From the main directory, start the FastAPI endpoint.
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

4. **Start the Frontend Application**
   In another terminal session, spin up the Streamlit UI.
   ```bash
   streamlit run frontend/app.py
   ```

## Demo Instructions
- Access Streamlit at `http://localhost:8501`.
- Enter queries such as "noise cancelling headphones" or "smart watch".
- Tweak the 'Search Mode' dropdown and interact with tabs to observe Rank Changes.
