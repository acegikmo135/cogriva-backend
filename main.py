"""
tutor_backend.py  v4  — OneSignal notifications + debug log endpoint
─────────────────────────────────────────────────────────────────────
Run (dev):
  uvicorn tutor_backend:app --reload --host 127.0.0.1 --port 8000

Run (prod):
  uvicorn tutor_backend:app --host 0.0.0.0 --port 8000 --loop uvloop --workers 1
"""

import os, sys, asyncio, hashlib, time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "AIzaSyATRyMOaF9isc_qvE2RIonhzk1fb_gFOjk")
SUPABASE_URL         = os.getenv("VITE_SUPABASE_URL", "https://betdygzhltshxsqzzqhq.supabase.co")
SUPABASE_ANON_KEY    = os.getenv("VITE_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJldGR5Z3pobHRzaHhzcXp6cWhxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU4MDQxODAsImV4cCI6MjA5MTM4MDE4MH0.O0zH_aXZolVVulSYoMnZA6Xx3DeLKuop7h-EEJcWxAA")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJldGR5Z3pobHRzaHhzcXp6cWhxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTgwNDE4MCwiZXhwIjoyMDkxMzgwMTgwfQ.V02ZQ-TFR5hMNLHAD4zAVYU-KZuNa9mcrq11yeunrJQ")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY", "")
ONESIGNAL_APP_ID     = os.getenv("VITE_ONESIGNAL_APP_ID", "")
ONESIGNAL_API_KEY    = os.getenv("ONESIGNAL_API_KEY", "")
PINECONE_INDEX       = "cognistruct-rag"
PINECONE_NAMESPACE   = "class8-science"
PINECONE_RAG_SCORE   = 0.75
# ══════════════════════════════════════════════════════════════════════════════

SIMILARITY_THRESHOLD = 0.80
GEMINI_MODEL         = "gemini-2.5-flash-lite"
JWT_CACHE_TTL        = 300

# ── Debug log buffer (last 500 entries, thread-safe via asyncio single-thread) ─
_LOG_BUFFER: deque = deque(maxlen=500)

def _log(level: str, msg: str, **ctx):
    """Write to stdout and append to the in-memory debug buffer."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    entry = {"ts": ts, "level": level.upper(), "msg": msg, **ctx}
    _LOG_BUFFER.append(entry)
    tag = f"[{level.upper()}]" if level.upper() != "INFO" else ""
    line = f"{tag} {msg}" if tag else msg
    print(line, flush=True)

# ── Supabase REST headers ─────────────────────────────────────────────────────
_SB_HEADERS = {
    "apikey":        SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}
_SB_ANON_HEADERS = {
    "apikey":        SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type":  "application/json",
}

# ── Static responses ──────────────────────────────────────────────────────────
import re as _re

def _norm(q: str) -> str:
    return _re.sub(r'[^\w\s]', '', q.strip().lower()).strip()

STATIC_RESPONSES: dict[str, str] = {
    'hi':                   "Hi there! I'm CogniStruct AI, your study tutor. Ask me anything about your chapter!",
    'hello':                "Hello! Ready to help you study. What would you like to know?",
    'hey':                  "Hey! What can I help you learn today?",
    'heyy':                 "Hey! What would you like to study today?",
    'bye':                  "Goodbye! Keep studying hard! 📚",
    'goodbye':              "See you later! Good luck with your studies!",
    'good bye':             "Goodbye! Come back anytime you need help studying.",
    'ok':                   "Got it! Ask me anything you'd like to know about your chapter.",
    'okay':                 "Alright! What would you like to learn?",
    'k':                    "Sure! What's your question?",
    'yes':                  "Great! What would you like to know?",
    'no':                   "No problem! Let me know if you have any questions.",
    'thanks':               "You're welcome! Let me know if you have more questions. 😊",
    'thank you':            "Happy to help! Any more questions?",
    'thank u':              "You're welcome! Keep it up!",
    'ty':                   "You're welcome!",
    'thx':                  "Anytime! Ask me more if you need help.",
    'cool':                 "Glad to hear that! Keep going — you're doing great!",
    'great':                "Awesome! Keep up the good work!",
    'nice':                 "Thanks! What else would you like to learn?",
    'wow':                  "I know, right! Learning is amazing. What's next?",
    'amazing':              "Thank you! Now let's keep learning. What's your next question?",
    'awesome':              "You're awesome too! Keep studying!",
    'good':                 "Good to hear! What else can I help you with?",
    'fine':                 "Alright! Let me know if you have any questions.",
    'sure':                 "Of course! What would you like to know?",
    'got it':               "Great! Let me know if anything is unclear.",
    'understood':           "Excellent! Keep up the good work!",
    'noted':                "Perfect! Ask me if you need any clarification.",
    'alright':              "Alright! Ask me anything about your chapter.",
    'hmm':                  "Take your time! Ask me whenever you're ready.",
    'lol':                  "Haha! Now, let's get back to studying. 😄",
    'haha':                 "😄 Okay, what would you like to learn next?",
    'who are you':          "I'm CogniStruct AI — your personal study tutor made by Manthan.",
    'what are you':         "I'm CogniStruct AI, an AI tutor built by Manthan to help students study smarter!",
    'what is your name':    "I'm CogniStruct AI! Made by Manthan to be your study companion.",
    'how are you':          "I'm doing great and fully charged to help you study! What would you like to know?",
    'how r u':              "I'm great, thanks for asking! What can I help you study today?",
    'who made you':         "I was made by Manthan!",
    'who created you':      "I was created by Manthan!",
    'who built you':        "Manthan built me!",
    'are you an ai':        "Yes, I'm CogniStruct AI — an AI tutor made by Manthan to help you study!",
    'are you real':         "I'm an AI tutor, but my help is very real! Ask me anything about your chapter.",
    'what can you do':      "I can explain concepts, answer questions, help with doubts, and guide you through your chapter. Just ask!",
    'help':                 "I'm here to help! Ask me any question about your chapter and I'll explain it simply.",
    'start':                "Let's go! What would you like to learn about your chapter?",
}

def get_static_response(q: str) -> str | None:
    normalized = _norm(q)
    if normalized in STATIC_RESPONSES:
        return STATIC_RESPONSES[normalized]
    if len(normalized.split()) <= 2 and len(normalized) < 20:
        return None
    return None

# ── In-process LRU ────────────────────────────────────────────────────────────
_LRU: dict[str, dict] = {}
_LRU_MAX = 512

def _lru_key(question: str, chapter_id: str) -> str:
    return hashlib.sha256(f"{chapter_id}::{question.lower().strip()}".encode()).hexdigest()

def lru_get(question: str, chapter_id: str) -> str | None:
    key = _lru_key(question, chapter_id)
    entry = _LRU.get(key)
    if entry:
        entry['hits'] += 1
        return entry['answer']
    return None

def lru_put(question: str, chapter_id: str, answer: str) -> None:
    key = _lru_key(question, chapter_id)
    _LRU[key] = {'answer': answer, 'hits': 1}
    if len(_LRU) > _LRU_MAX:
        evict_key = min(_LRU, key=lambda k: _LRU[k]['hits'])
        _LRU.pop(evict_key, None)

# ── JWT cache ─────────────────────────────────────────────────────────────────
_JWT: dict[str, tuple[str, float]] = {}

def _jwt_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def jwt_cache_get(token: str) -> str | None:
    h = _jwt_hash(token)
    entry = _JWT.get(h)
    if entry and time.monotonic() < entry[1]:
        return entry[0]
    _JWT.pop(h, None)
    return None

def jwt_cache_put(token: str, user_id: str) -> None:
    _JWT[_jwt_hash(token)] = (user_id, time.monotonic() + JWT_CACHE_TTL)
    if len(_JWT) > 1024:
        now = time.monotonic()
        stale = [k for k, v in _JWT.items() if now >= v[1]]
        for k in stale:
            _JWT.pop(k, None)

# ── Validate dependencies ─────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    sys.exit("Missing: pip install fastapi uvicorn")

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    from fastembed import TextEmbedding
except ImportError:
    sys.exit("Missing: pip install fastembed")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("Missing: pip install google-genai")

try:
    from pinecone import Pinecone as PineconeClient
except ImportError:
    sys.exit("Missing: pip install pinecone")

# ── Startup ───────────────────────────────────────────────────────────────────
_log("INFO", "Loading all-MiniLM-L6-v2 (ONNX)…")
encoder       = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

_pinecone_index = None
if PINECONE_API_KEY:
    try:
        _pc = PineconeClient(api_key=PINECONE_API_KEY)
        _pinecone_index = _pc.Index(PINECONE_INDEX)
        _log("INFO", "Pinecone RAG index connected.")
    except Exception as _e:
        _log("WARN", f"Pinecone init failed: {_e} — RAG disabled")

_encoder_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="onnx")

_log("INFO", "Pre-warming encoder…")
_ = list(encoder.embed(["warmup sentence for onnx runtime initialisation"]))
_log("INFO", "Ready ✅")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="CogniStruct Tutor Backend", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

_http: httpx.AsyncClient = None

_CLEANUP_INTERVAL     = 3600
_CLEANUP_MIN_AGE_DAYS = 7
_CLEANUP_HIT_THRESHOLD = 2

async def _cache_cleanup_loop():
    await asyncio.sleep(60)
    while True:
        try:
            cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - _CLEANUP_MIN_AGE_DAYS * 86400),
            )
            r = await _http.delete(
                f"{SUPABASE_URL}/rest/v1/tutor_cache",
                headers={**_SB_HEADERS, "Prefer": "return=representation"},
                params={
                    "hit_count": f"lt.{_CLEANUP_HIT_THRESHOLD}",
                    "created_at": f"lt.{cutoff}",
                },
            )
            deleted = len(r.json()) if r.status_code == 200 else 0
            if deleted:
                _log("INFO", f"[CLEANUP] deleted {deleted} low-hit cache entries")
        except Exception as e:
            _log("ERROR", f"[CLEANUP] {e}")
        await asyncio.sleep(_CLEANUP_INTERVAL)


@app.on_event("startup")
async def startup():
    global _http
    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    asyncio.create_task(_cache_cleanup_loop())

@app.on_event("shutdown")
async def shutdown():
    await _http.aclose()

LATEX = "Use LaTeX for ALL math expressions in single dollar signs (e.g. $x^2$). Use **bold** for key terms."

# ── Models ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role:    str
    content: str

class AskRequest(BaseModel):
    question:   str
    chapter_id: str
    context:    str
    history:    list[Message]
    token:      str

class AskResponse(BaseModel):
    answer:       str
    cached:       bool
    similarity:   float | None = None
    rate_limited: bool = False

class NotificationRequest(BaseModel):
    target_user_id:     str
    title:              str
    message:            str
    send_after_seconds: int | None = None
    token:              str

# ── Helpers ───────────────────────────────────────────────────────────────────

async def verify_jwt(token: str) -> str:
    if not SUPABASE_ANON_KEY:
        return "dev-no-auth"
    cached = jwt_cache_get(token)
    if cached:
        return cached
    try:
        r = await _http.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={**_SB_ANON_HEADERS, "Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        user_id = r.json().get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Unauthorized")
        jwt_cache_put(token, user_id)
        return user_id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _embed_sync(text: str) -> list[float]:
    return list(encoder.embed([text]))[0].tolist()

async def embed_async(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_encoder_pool, _embed_sync, text)


async def vec_search(q_vec: list[float], chapter_id: str) -> dict | None:
    try:
        r = await _http.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_tutor_cache",
            headers=_SB_HEADERS,
            json={
                "query_embedding":      q_vec,
                "filter_chapter_id":   chapter_id,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "match_count":          1,
            },
        )
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except Exception as e:
        _log("ERROR", f"vec-search: {e}")
        return None


async def bump_hit(cache_id: str) -> None:
    try:
        await _http.post(
            f"{SUPABASE_URL}/rest/v1/rpc/increment_cache_hit",
            headers=_SB_HEADERS,
            json={"cache_id": cache_id},
        )
    except Exception:
        pass


async def store_cache(chapter_id: str, question: str, answer: str, embedding: list[float]) -> None:
    try:
        await _http.post(
            f"{SUPABASE_URL}/rest/v1/tutor_cache",
            headers=_SB_HEADERS,
            json={"chapter_id": chapter_id, "question": question,
                  "answer": answer, "embedding": embedding},
        )
    except Exception as e:
        _log("ERROR", f"store-cache: {e}")


def _subject_hint(context: str) -> str:
    subject = ""
    for part in context.split("|"):
        if "Subject:" in part:
            subject = part.split(":", 1)[1].strip().lower()
            break
    if   "math"    in subject: return "Solve maths step by step. Show each step clearly."
    elif "physics" in subject: return "Concept first, then formula, then a simple example."
    elif "chem"    in subject: return "Focus on what happens and why, using everyday examples."
    elif "bio"     in subject or "science" in subject: return "Use body or nature examples."
    elif "hist"    in subject: return "Focus on who, what, when, why. Keep dates simple."
    elif "geo"     in subject: return "Relate to real places the student has seen."
    elif "civic"   in subject or "political" in subject: return "Use real Indian government examples."
    elif "eco"     in subject: return "Use market/shop examples from daily life."
    else:                      return "Give a clear, direct explanation with a simple real-life example."


_GLOBAL_LIMIT: dict = {}
_GLOBAL_LIMIT_TTL = 300

async def _fetch_global_token_limit() -> int:
    try:
        r = await _http.get(
            f"{SUPABASE_URL}/rest/v1/admin_settings",
            headers=_SB_HEADERS,
            params={"key": "eq.app_settings", "select": "value"},
        )
        if r.status_code == 200:
            rows = r.json()
            if rows:
                val = (rows[0].get("value") or {})
                return int(val.get("rate_limits", {}).get("chatbot_daily_tokens", 5000))
    except Exception as e:
        _log("ERROR", f"global-limit-fetch: {e}")
    return 5000

async def get_global_token_limit() -> int:
    now = time.monotonic()
    if _GLOBAL_LIMIT.get("ts", 0) + _GLOBAL_LIMIT_TTL > now:
        return _GLOBAL_LIMIT["limit"]
    limit = await _fetch_global_token_limit()
    _GLOBAL_LIMIT["limit"] = limit
    _GLOBAL_LIMIT["ts"]    = now
    _log("INFO", f"global-limit: {limit} tokens/day")
    return limit

async def check_rate_limit(user_id: str) -> tuple[bool, int, int]:
    try:
        daily_limit, usage_res = await asyncio.gather(
            get_global_token_limit(),
            _http.post(f"{SUPABASE_URL}/rest/v1/rpc/get_user_daily_tokens",
                       headers=_SB_HEADERS, json={"p_user_id": user_id}),
            return_exceptions=True,
        )
        if isinstance(daily_limit, Exception):
            daily_limit = 5000
        tokens_used = 0
        if not isinstance(usage_res, Exception) and usage_res.status_code == 200:
            tokens_used = usage_res.json() or 0
        return tokens_used >= daily_limit, tokens_used, daily_limit
    except Exception as e:
        _log("ERROR", f"rate-limit: {e}")
        return False, 0, 5000


async def log_usage(user_id: str, tokens_in: int, tokens_out: int) -> None:
    try:
        await _http.post(
            f"{SUPABASE_URL}/rest/v1/usage_logs",
            headers={**_SB_HEADERS, "Prefer": "return=minimal"},
            json={"user_id": user_id, "feature": "doubt_solver",
                  "tokens_in": tokens_in, "tokens_out": tokens_out},
        )
    except Exception as e:
        _log("ERROR", f"log-usage: {e}")


_RAG_AVAILABLE = [
    ("science", "crop production"),
]

def _has_rag_data(context: str) -> bool:
    ctx = context.lower()
    return any(subj in ctx and chap in ctx for subj, chap in _RAG_AVAILABLE)


async def pinecone_rag_search(question: str, context: str) -> str | None:
    if _pinecone_index is None:
        return None
    if not _has_rag_data(context):
        _log("INFO", "RAG: no dataset for this chapter — skipping")
        return None
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _pinecone_index.search(
                namespace=PINECONE_NAMESPACE,
                query={"inputs": {"text": question}, "top_k": 1},
            ),
        )
        hits = results.get("result", {}).get("hits", [])
        if hits and hits[0].get("_score", 0) >= PINECONE_RAG_SCORE:
            h     = hits[0]
            chunk = h["fields"].get("text", "")
            topic = h["fields"].get("topic", h["_id"])
            _log("INFO", f"RAG: score={h['_score']:.3f} topic={topic}")
            return chunk
        _log("INFO", f"RAG: no hit above threshold ({PINECONE_RAG_SCORE})")
    except Exception as e:
        _log("ERROR", f"pinecone-search: {e}")
    return None


async def call_gemini(question: str, context: str, history: list[Message], rag_chunk: str | None = None) -> str:
    if rag_chunk:
        rag_block = f"""

TEXTBOOK CONTENT:
\"\"\"
{rag_chunk}
\"\"\"
RULE: Answer the student's question using ONLY the textbook content above. \
Do not add anything outside it. Keep the answer simple and short."""
    else:
        rag_block = ""

    system_instruction = f"""You are an AI tutor named CogniStruct, made by Manthan.
RULES:
1. Answer ONLY what was asked. Nothing extra.
2. Use the simplest words possible, as if explaining to a 12-year-old.
3. Keep answers SHORT — 2–4 sentences for simple questions.
4. No introductions or summaries. Answer directly.
5. Never say "Great question!" or "Sure!" — just answer.
6. {_subject_hint(context)}
7. {LATEX}
8. If asked who made you: always say "I was made by Manthan." Never mention Google or Gemini.{rag_block}"""

    ctx_history = [
        {"role": "user",  "parts": [{"text": f"[Student profile]: {context}"}]},
        {"role": "model", "parts": [{"text": "Understood."}]},
    ]
    gemini_history = ctx_history + [
        {"role": m.role if m.role == "user" else "model",
         "parts": [{"text": m.content[:800]}]}
        for m in history[-20:]
        if m.role in ("user", "model") and m.content.strip()
    ]
    if gemini_history and gemini_history[-1]["role"] == "user":
        last = gemini_history.pop()["parts"][0]["text"]
    else:
        last = question

    chat = gemini_client.aio.chats.create(
        model=GEMINI_MODEL,
        config=genai_types.GenerateContentConfig(system_instruction=system_instruction),
        history=gemini_history,
    )
    resp = await chat.send_message(last)
    tokens_in  = getattr(resp.usage_metadata, "prompt_token_count",     0) or 0
    tokens_out = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
    return (resp.text or "").strip(), tokens_in, tokens_out


# ── /ask ──────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    t0 = time.monotonic()
    user_id = await verify_jwt(req.token)
    question = req.question.strip()[:800]

    static = get_static_response(question)
    if static:
        _log("INFO", f"[STATIC] {int((time.monotonic()-t0)*1000)}ms | {question[:50]}")
        return AskResponse(answer=static, cached=True)

    cached = lru_get(question, req.chapter_id)
    if cached:
        _log("INFO", f"[MEM] {int((time.monotonic()-t0)*1000)}ms | {question[:50]}")
        return AskResponse(answer=cached, cached=True, similarity=1.0)

    q_vec = await embed_async(question)
    hit   = await vec_search(q_vec, req.chapter_id)

    if hit:
        asyncio.create_task(bump_hit(hit["id"]))
        lru_put(question, req.chapter_id, hit["answer"])
        _log("INFO", f"[VEC] {int((time.monotonic()-t0)*1000)}ms sim={hit['similarity']:.3f} | {question[:50]}")
        return AskResponse(answer=hit["answer"], cached=True, similarity=hit["similarity"])

    is_limited, tokens_used, daily_limit = await check_rate_limit(user_id)
    if is_limited:
        _log("WARN", f"[RATE-LIMIT] user={user_id[:8]} used={tokens_used}/{daily_limit}")
        return AskResponse(
            answer=(
                f"🚦 You've used all **{daily_limit:,} tokens** for today.\n\n"
                "Your limit resets at **midnight UTC**. Come back tomorrow!\n\n"
                "💡 Good news: answers from cache are always free and don't count."
            ),
            cached=False,
            rate_limited=True,
        )

    rag_chunk = await pinecone_rag_search(question, req.context)
    answer, tokens_in, tokens_out = await call_gemini(question, req.context, req.history, rag_chunk)
    lru_put(question, req.chapter_id, answer)
    asyncio.create_task(store_cache(req.chapter_id, question, answer, q_vec))
    asyncio.create_task(log_usage(user_id, tokens_in, tokens_out))
    _log("INFO", f"[MISS] {int((time.monotonic()-t0)*1000)}ms tok={tokens_in}in/{tokens_out}out | {question[:50]}")
    return AskResponse(answer=answer, cached=False)


_UUID_RE = _re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    _re.IGNORECASE,
)


async def _send_onesignal(target_user_id: str, title: str, message: str) -> dict:
    """Send via OneSignal REST API using the user's Supabase UUID as external_id."""
    if not ONESIGNAL_APP_ID or not ONESIGNAL_API_KEY:
        _log("WARN", "OneSignal: ONESIGNAL_APP_ID / ONESIGNAL_API_KEY not set — skipped")
        return {"status": "skipped", "reason": "OneSignal not configured"}

    r = await _http.post(
        "https://api.onesignal.com/notifications",
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Key {ONESIGNAL_API_KEY}",
        },
        json={
            "app_id":          ONESIGNAL_APP_ID,
            "target_channel":  "push",
            "include_aliases": {"external_id": [target_user_id]},
            "headings":        {"en": title[:100]},
            "contents":        {"en": message[:300]},
        },
    )
    data = r.json()
    if r.status_code == 200:
        _log("INFO", f"OneSignal ✅ HTTP {r.status_code} → {data}")
    else:
        _log("ERROR", f"OneSignal ✗ HTTP {r.status_code} → {data}")
    return data


# ── /send-notification ────────────────────────────────────────────────────────
@app.post("/send-notification")
async def send_notification(req: NotificationRequest):
    await verify_jwt(req.token)

    _log("INFO", f"/send-notification target={req.target_user_id!r} title={req.title!r} delay={req.send_after_seconds}s")

    if not _UUID_RE.match(req.target_user_id):
        _log("WARN", f"send-notification: invalid UUID {req.target_user_id!r}")
        raise HTTPException(status_code=400, detail="Invalid target_user_id")

    if req.send_after_seconds and req.send_after_seconds > 0:
        # Fire-and-forget after delay — keeps the HTTP response instant
        async def _delayed():
            await asyncio.sleep(req.send_after_seconds)
            await _send_onesignal(req.target_user_id, req.title, req.message)

        asyncio.create_task(_delayed())
        _log("INFO", f"send-notification: scheduled in {req.send_after_seconds}s")
        return {"status": "scheduled", "send_after_seconds": req.send_after_seconds}

    result = await _send_onesignal(req.target_user_id, req.title, req.message)
    return result


# ── /logs — debug log viewer ──────────────────────────────────────────────────
@app.get("/logs")
async def get_logs(level: str | None = None, limit: int = 100, q: str | None = None):
    """
    Returns recent backend log entries.
    Query params:
      level  — filter by level: INFO, WARN, ERROR
      limit  — max entries to return (default 100, max 500)
      q      — substring search in message
    """
    limit = min(max(1, limit), 500)
    logs  = list(_LOG_BUFFER)

    if level:
        logs = [e for e in logs if e["level"] == level.upper()]
    if q:
        logs = [e for e in logs if q.lower() in e["msg"].lower()]

    return {
        "total_in_buffer": len(_LOG_BUFFER),
        "returned":        len(logs[-limit:]),
        "logs":            logs[-limit:],
    }


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":            "ok",
        "lru_size":          len(_LRU),
        "jwt_cache_size":    len(_JWT),
        "log_buffer_size":   len(_LOG_BUFFER),
        "onesignal_ready":   bool(ONESIGNAL_APP_ID and ONESIGNAL_API_KEY),
        "pinecone_ready":    _pinecone_index is not None,
    }
