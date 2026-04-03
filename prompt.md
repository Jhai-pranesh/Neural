You are a senior full-stack + ML engineer. Build a complete, production-quality mini project called **NeuralSearch Prime (Demo Version)** based on the following requirements.

---

## 🎯 OBJECTIVE

Build a **two-stage e-commerce search system** with:

1. Retrieval stage (use external API as backend)
2. Reranking stage (local lightweight scoring)
3. Clean, professional UI
4. Robust error handling
5. Deployed-ready structure

This is a **fully working demo system**, not a research prototype.

---

## 🧠 SYSTEM ARCHITECTURE

### Pipeline:

User Query → FastAPI Backend → Retrieval → Reranking → UI Display

---

## ⚙️ BACKEND REQUIREMENTS (FastAPI)

### Endpoint:

POST /search

Request:
{
"query": "string",
"top_k": int
}

Response:
{
"results": [
{
"title": "...",
"score": float,
"original_rank": int,
"reranked_rank": int,
"rank_change": int
}
],
"mode": "lexical | hybrid | neural"
}

---

### Stage 1: Retrieval

* Use a placeholder external search API (make an env file i will provide the serpAPi for amazon search)
* Return top 50 results
* Each result must include:

  * title
  * description
  * id

---

### Stage 2: Reranking

Implement a lightweight reranker:

1. Use sentence-transformers (all-MiniLM-L6-v2)

2. Compute cosine similarity between:
   query ↔ (title + description)

3. Also compute keyword overlap score

Final score:
score = 0.7 * semantic_similarity + 0.3 * keyword_overlap

---

### Rank Processing:

* Store original rank
* Sort by new score
* Compute:
  rank_change = original_rank - new_rank

---

### Fallback Logic:

* If reranker fails → return Stage 1 results
* Include flag in response

---

### Error Handling:

* Empty query → return error message
* API failure → fallback + warning
* Timeout → handled gracefully

---

## 🎨 FRONTEND REQUIREMENTS (Streamlit)

### Layout:

* Wide layout
* Top header: "NeuralSearch Prime"
* Subheading: "Hybrid + Neural Reranked Search"

---

### Input Section:

* Search bar (centered, large)
* Dropdown:

  * Lexical
  * Hybrid
  * Neural Reranked
* Top-K slider (10–50)
* Search button

---

### UX Feedback:

* Spinner:
  "Retrieving results..."
  "Reranking results..."

---

### Results Display:

Use **card-style UI** (VERY IMPORTANT):
Each result shows:

* Title (bold)
* Score (formatted)
* Rank change:

  * ↑ green if improved
  * ↓ red if dropped
* Original vs New rank

---

### Tabs:

* Lexical Results
* Hybrid Results
* Neural Reranked Results

---

### Styling:

* Use clean spacing
* Rounded containers
* Subtle shadows
* Highlight top 3 results

---

## 🛡️ ROBUSTNESS

* No blank screens EVER
* Show:

  * "No results found"
  * "Error occurred, showing fallback results"
* Handle invalid inputs gracefully

---

## 🚀 DEPLOYMENT READY

* Provide:

  * requirements.txt
  * run instructions
* Structure:
  backend/
  frontend/
  README.md

---

## 📄 README CONTENT

Include:

* Project overview
* Architecture diagram (text-based is fine)
* Features:

  * Hybrid retrieval
  * Neural reranking
  * Rank comparison
* Setup instructions
* Known limitations

---

## ⚡ EXTRA (if time permits)

* Add latency measurement (display ms)
* Add simple logging
* Add mock dataset if API unavailable

---

## 🧩 IMPORTANT CONSTRAINTS

* Code must be clean and modular
* No placeholder UI — everything must work
* Must run locally without GPU
* Must not crash under bad input
* Prioritize working demo over complexity

---

## OUTPUT FORMAT

Generate:

1. Full backend code (FastAPI)
2. Full frontend code (Streamlit)
3. requirements.txt
4. README.md
5. Clear run instructions

Ensure everything works end-to-end.
