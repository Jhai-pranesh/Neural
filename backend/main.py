import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from serpapi import GoogleSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(title="NeuralSearch Prime API")

# Lazy-load sentence-transformers so Uvicorn binds to $PORT immediately (Render port scan).
MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None
_st_util = None
_embedding_lock = threading.Lock()
_embedding_failed = False


def get_sentence_model():
    """Load MiniLM on first use; avoids blocking startup while downloading/loading weights."""
    global _embedding_model, _st_util, _embedding_failed
    if _embedding_failed:
        return None
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        if _embedding_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer, util as st_util_module

            logger.info("Loading sentence transformer: %s (first search may take a minute on cold start)", MODEL_NAME)
            _embedding_model = SentenceTransformer(MODEL_NAME)
            _st_util = st_util_module
            logger.info("Loaded sentence transformer: %s", MODEL_NAME)
        except Exception as e:
            _embedding_failed = True
            logger.warning("Sentence transformer failed to load: %s", e)
            return None
    return _embedding_model

_cors_extra = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Redis (optional): only connect when REDIS_URL is set ---
CACHE_TTL_SEC = 1800
RECENT_MAX = 4
REDIS_RECENT_KEY = "search:recent_keys"

redis_client = None
_redis_url = os.getenv("REDIS_URL", "").strip()
if _redis_url:
    try:
        import redis

        redis_client = redis.from_url(_redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connected at %s", _redis_url)
    except Exception as e:
        redis_client = None
        logger.warning("Redis unavailable (caching disabled): %s", e)
else:
    logger.info("REDIS_URL not set; caching disabled")


def normalize_query(q: str) -> str:
    return q.strip().lower()


def redis_cache_key(mode: str, query: str) -> str:
    return f"search:cache:{mode}:{normalize_query(query)}"


def cache_get(mode: str, query: str) -> Optional[dict]:
    if not redis_client:
        return None
    key = redis_cache_key(mode, query)
    raw = redis_client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _recent_token(mode: str, query: str) -> str:
    return json.dumps({"mode": mode, "q": normalize_query(query)}, sort_keys=True)


def triple_cache_key(nq: str, top_k: int) -> str:
    return f"search:cache:triple:{nq}:{top_k}"


def _recent_token_triple(query: str, top_k: int) -> str:
    return json.dumps(
        {"mode": "triple", "q": normalize_query(query), "k": top_k},
        sort_keys=True,
    )


def _evict_dropped_recent_entries(before: set, after: set) -> None:
    if not redis_client:
        return
    for old_t in before:
        if old_t not in after:
            try:
                d = json.loads(old_t)
                if d.get("mode") == "triple":
                    redis_client.delete(
                        triple_cache_key(str(d["q"]), int(d["k"]))
                    )
                else:
                    redis_client.delete(redis_cache_key(d["mode"], d["q"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue


def cache_set(mode: str, query: str, payload: dict) -> None:
    if not redis_client:
        return
    key = redis_cache_key(mode, query)
    redis_client.setex(key, CACHE_TTL_SEC, json.dumps(payload))

    token = _recent_token(mode, query)
    before = set(redis_client.lrange(REDIS_RECENT_KEY, 0, -1))
    redis_client.lrem(REDIS_RECENT_KEY, 0, token)
    redis_client.lpush(REDIS_RECENT_KEY, token)
    redis_client.ltrim(REDIS_RECENT_KEY, 0, RECENT_MAX - 1)
    after = set(redis_client.lrange(REDIS_RECENT_KEY, 0, -1))
    _evict_dropped_recent_entries(before, after)


def cache_get_triple(query: str, top_k: int) -> Optional[dict]:
    if not redis_client:
        return None
    key = triple_cache_key(normalize_query(query), top_k)
    raw = redis_client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set_triple(query: str, top_k: int, payload: dict) -> None:
    if not redis_client:
        return
    key = triple_cache_key(normalize_query(query), top_k)
    redis_client.setex(key, CACHE_TTL_SEC, json.dumps(payload))

    token = _recent_token_triple(query, top_k)
    before = set(redis_client.lrange(REDIS_RECENT_KEY, 0, -1))
    redis_client.lrem(REDIS_RECENT_KEY, 0, token)
    redis_client.lpush(REDIS_RECENT_KEY, token)
    redis_client.ltrim(REDIS_RECENT_KEY, 0, RECENT_MAX - 1)
    after = set(redis_client.lrange(REDIS_RECENT_KEY, 0, -1))
    _evict_dropped_recent_entries(before, after)


SearchMode = Literal["lexical", "neural", "hybrid"]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: SearchMode = "neural"


class SearchResult(BaseModel):
    id: str
    title: str
    description: str
    score: float
    original_rank: int
    reranked_rank: int
    rank_change: int
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    mode: str
    latency_ms: float
    cached: bool = False
    search_fingerprint: str = ""


class TripleSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class TripleSearchResponse(BaseModel):
    lexical: List[SearchResult]
    neural: List[SearchResult]
    hybrid: List[SearchResult]
    latency_ms: float
    cached: bool = False
    search_fingerprint: str = ""


class NeuralExplainRequest(BaseModel):
    query: str
    top_three: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top 3 hits: title, description, score",
    )


class NeuralExplainResponse(BaseModel):
    explanation: str
    error: Optional[str] = None


def get_serpapi_results(query: str, top_k: int) -> List[dict]:
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key or api_key == "your_key_here":
        raise Exception("SERPAPI_KEY not set. Please add your key to the .env file.")

    amazon_params = {
        "engine": "amazon",
        "amazon_domain": "amazon.com",
        "k": query,
        "api_key": api_key,
    }
    try:
        logger.info("Trying Amazon SerpAPI engine for query: %s", query)
        search = GoogleSearch(amazon_params)
        results = search.get_dict()
        if "error" not in results:
            amazon_results = results.get("organic_results", results.get("amazon_results", []))
            if amazon_results:
                logger.info("Amazon engine returned %s results", len(amazon_results))
                return amazon_results
            logger.warning("Amazon engine returned no results, falling back to Google Shopping")
        else:
            logger.warning("Amazon engine error: %s, falling back", results["error"])
    except Exception as e:
        logger.warning("Amazon engine failed: %s, falling back to Google Shopping", e)

    shopping_params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": api_key,
        "num": min(top_k, 50),
        "gl": "us",
        "hl": "en",
    }
    try:
        logger.info("Trying Google Shopping engine for query: %s", query)
        search = GoogleSearch(shopping_params)
        results = search.get_dict()
        if "error" in results:
            raise Exception(f"Google Shopping error: {results['error']}")
        shopping_results = results.get("shopping_results", [])
        logger.info("Google Shopping returned %s results", len(shopping_results))
        return shopping_results
    except Exception as e:
        raise Exception(f"All SerpAPI engines failed: {e}") from e


def compute_keyword_overlap(query: str, text: str) -> float:
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words.intersection(t_words)) / len(q_words)


def build_results_for_mode(
    extracted: List[dict],
    mode: str,
    cos_scores: Optional[Any],
    docs_text: List[str],
    query: str,
    top_k: int,
) -> List[SearchResult]:
    """Produce ranked list and scores for lexical / neural / hybrid."""
    items: List[dict] = []
    for i, e in enumerate(extracted):
        doc = docs_text[i]
        kw = compute_keyword_overlap(query, doc)
        sem = float(cos_scores[i].item()) if cos_scores is not None else 0.0

        if mode == "lexical":
            final = kw
        elif mode == "neural":
            final = 0.7 * sem + 0.3 * kw
        else:  # hybrid
            final = 0.5 * sem + 0.5 * kw

        items.append(
            {
                **e,
                "semantic_score": sem,
                "keyword_score": kw,
                "score": final,
            }
        )

    items.sort(key=lambda x: (-x["score"], x["original_rank"]))

    out: List[SearchResult] = []
    for new_rank, item in enumerate(items):
        rr = new_rank + 1
        out.append(
            SearchResult(
                id=item["id"],
                title=item["title"],
                description=item["description"],
                score=round(item["score"], 4),
                original_rank=item["original_rank"],
                reranked_rank=rr,
                rank_change=item["original_rank"] - rr,
                semantic_score=round(item["semantic_score"], 4),
                keyword_score=round(item["keyword_score"], 4),
            )
        )
    return out[:top_k]


@app.post("/search/triple", response_model=TripleSearchResponse)
async def search_triple(req: TripleSearchRequest):
    """One SerpAPI fetch; return lexical, neural, and hybrid rankings side-by-side."""
    start_time = time.time()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    fp = f"triple:{normalize_query(req.query)}:{req.top_k}"
    cached = cache_get_triple(req.query, req.top_k)
    if cached is not None:
        cached["cached"] = True
        cached["search_fingerprint"] = fp
        return TripleSearchResponse(**cached)

    try:
        raw_results = get_serpapi_results(req.query, 50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    extracted: List[dict] = []
    for idx, r in enumerate(raw_results):
        title = r.get("title", "")
        description = (
            r.get("description")
            or r.get("snippet")
            or r.get("source")
            or r.get("price")
            or ""
        )
        item_id = r.get("asin") or r.get("product_id") or str(idx)
        if not title:
            continue
        extracted.append(
            {
                "id": str(item_id),
                "title": str(title),
                "description": str(description),
                "original_rank": idx + 1,
            }
        )

    extracted = extracted[:50]

    if not extracted:
        latency = (time.time() - start_time) * 1000
        body = {
            "lexical": [],
            "neural": [],
            "hybrid": [],
            "latency_ms": latency,
            "cached": False,
            "search_fingerprint": fp,
        }
        return TripleSearchResponse(**body)

    m = get_sentence_model()
    if m is None or _st_util is None:
        fb: List[SearchResult] = []
        for e in extracted[: req.top_k]:
            fb.append(
                SearchResult(
                    id=e["id"],
                    title=e["title"],
                    description=e["description"],
                    score=1.0,
                    original_rank=e["original_rank"],
                    reranked_rank=e["original_rank"],
                    rank_change=0,
                    semantic_score=None,
                    keyword_score=None,
                )
            )
        latency = (time.time() - start_time) * 1000
        payload = {
            "lexical": [r.model_dump() for r in fb],
            "neural": [r.model_dump() for r in fb],
            "hybrid": [r.model_dump() for r in fb],
            "latency_ms": latency,
            "cached": False,
            "search_fingerprint": fp,
        }
        cache_set_triple(req.query, req.top_k, payload)
        return TripleSearchResponse(**payload)

    docs_text = [f"{e['title']} {e['description']}" for e in extracted]
    query_emb = m.encode(req.query, convert_to_tensor=True)
    docs_emb = m.encode(docs_text, convert_to_tensor=True)
    cos_scores = _st_util.cos_sim(query_emb, docs_emb)[0]

    lexical_r = build_results_for_mode(
        extracted, "lexical", cos_scores, docs_text, req.query, req.top_k
    )
    neural_r = build_results_for_mode(
        extracted, "neural", cos_scores, docs_text, req.query, req.top_k
    )
    hybrid_r = build_results_for_mode(
        extracted, "hybrid", cos_scores, docs_text, req.query, req.top_k
    )

    latency = (time.time() - start_time) * 1000
    payload = {
        "lexical": [r.model_dump() for r in lexical_r],
        "neural": [r.model_dump() for r in neural_r],
        "hybrid": [r.model_dump() for r in hybrid_r],
        "latency_ms": latency,
        "cached": False,
        "search_fingerprint": fp,
    }
    cache_set_triple(req.query, req.top_k, payload)
    payload["search_fingerprint"] = fp
    return TripleSearchResponse(**payload)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    start_time = time.time()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    fp = f"{req.mode}:{normalize_query(req.query)}:{req.top_k}"
    cached = cache_get(req.mode, req.query)
    if cached is not None:
        cached["cached"] = True
        cached["search_fingerprint"] = fp
        return SearchResponse(**cached)

    try:
        raw_results = get_serpapi_results(req.query, 50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    extracted: List[dict] = []
    for idx, r in enumerate(raw_results):
        title = r.get("title", "")
        description = (
            r.get("description")
            or r.get("snippet")
            or r.get("source")
            or r.get("price")
            or ""
        )
        item_id = r.get("asin") or r.get("product_id") or str(idx)
        if not title:
            continue
        extracted.append(
            {
                "id": str(item_id),
                "title": str(title),
                "description": str(description),
                "original_rank": idx + 1,
            }
        )

    extracted = extracted[:50]

    if not extracted:
        latency = (time.time() - start_time) * 1000
        body = {
            "results": [],
            "mode": req.mode,
            "latency_ms": latency,
            "cached": False,
            "search_fingerprint": fp,
        }
        return SearchResponse(**body)

    m = get_sentence_model()
    if m is None or _st_util is None:
        results: List[SearchResult] = []
        for e in extracted[: req.top_k]:
            results.append(
                SearchResult(
                    id=e["id"],
                    title=e["title"],
                    description=e["description"],
                    score=1.0,
                    original_rank=e["original_rank"],
                    reranked_rank=e["original_rank"],
                    rank_change=0,
                    semantic_score=None,
                    keyword_score=None,
                )
            )
        latency = (time.time() - start_time) * 1000
        payload = {
            "results": [r.model_dump() for r in results],
            "mode": f"{req.mode} (fallback)",
            "latency_ms": latency,
            "cached": False,
            "search_fingerprint": fp,
        }
        cache_set(req.mode, req.query, payload)
        return SearchResponse(**payload)

    docs_text = [f"{e['title']} {e['description']}" for e in extracted]
    query_emb = m.encode(req.query, convert_to_tensor=True)
    docs_emb = m.encode(docs_text, convert_to_tensor=True)
    cos_scores = _st_util.cos_sim(query_emb, docs_emb)[0]

    final_results = build_results_for_mode(
        extracted, req.mode, cos_scores, docs_text, req.query, req.top_k
    )

    latency = (time.time() - start_time) * 1000
    payload = {
        "results": [r.model_dump() for r in final_results],
        "mode": req.mode,
        "latency_ms": latency,
        "cached": False,
        "search_fingerprint": fp,
    }
    cache_set(req.mode, req.query, payload)
    payload["search_fingerprint"] = fp
    return SearchResponse(**payload)


@app.post("/neural-explain", response_model=NeuralExplainResponse)
async def neural_explain(body: NeuralExplainRequest):
    """LLM rationale for why the top 3 neural results were retrieved (Groq)."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return NeuralExplainResponse(
            explanation="",
            error="GROQ_API_KEY is not set in the environment.",
        )

    if not body.top_three:
        return NeuralExplainResponse(explanation="", error="No results to explain.")

    model_id = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    lines = []
    for i, item in enumerate(body.top_three[:3], start=1):
        title = item.get("title", "")
        desc = (item.get("description") or "")[:400]
        score = item.get("score", "")
        lines.append(f"{i}. {title}\n   Snippet: {desc}\n   Combined score: {score}")

    user_prompt = (
        f"User query: {body.query}\n\n"
        f"These are the top 3 product results after neural reranking "
        f"(semantic similarity + keyword overlap):\n\n"
        + "\n\n".join(lines)
        + "\n\nIn 3–5 short bullet points, explain why these three listings are plausible "
        "top matches for the query from a shopper's perspective. "
        "Mention query terms and semantic fit. Do not invent product facts not in the snippets."
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a concise search-quality analyst. Be factual and brief.",
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            )
        r.raise_for_status()
        data = r.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return NeuralExplainResponse(explanation=text)
    except Exception as e:
        logger.exception("Groq explain failed")
        return NeuralExplainResponse(explanation="", error=str(e))


@app.get("/health")
async def health():
    # Do not call get_sentence_model() here — that would load weights and slow health checks.
    return {
        "redis": bool(redis_client),
        "embedding_model_loaded": _embedding_model is not None,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }
