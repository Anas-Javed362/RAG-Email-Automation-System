# RAG-Based Email Automation System v2.0

<div align="center">

🤖 **Production-grade AI backend** that classifies incoming emails and generates context-aware responses using **Retrieval-Augmented Generation (RAG)**.

Built with FastAPI · LangChain · FAISS · sentence-transformers · SQLite · Docker

</div>

---

## 🏗️ Architecture

```
Incoming Email
     │
     ▼
Middleware (Request ID)
     │
     ▼
EmailService (async orchestrator)
     │
     ├─► EmailCleaner ─────────── HTML strip, signature removal, whitespace normalization
     │
     ├─► Cache Check ─────────── SHA-256 hash → memory or Redis cache
     │
     ├─► Classifier ──────────── Rule-based keywords + LLM zero-shot (parallel signals)
     │
     ├─► Retriever (concurrent)
     │     ├─► FAISS search ──── Top-k semantic similarity (cosine)
     │     └─► Thread History ── Past messages for same thread_id
     │
     ├─► Generator ───────────── Versioned prompt (v1/v2) + retry + LLM self-eval
     │
     ├─► Confidence Fusion ───── 0.4 × cls + 0.4 × sim + 0.2 × llm_self
     │
     └─► DB Persist + Cache ──── SQLite (Email + ThreadMessage rows) + FAISS vector
```

---

## 📂 Project Structure

```
rag based email/
├── app/
│   ├── main.py                     # FastAPI app + lifespan
│   ├── core/
│   │   ├── logger.py               # RequestContext + structured logging
│   │   └── middleware.py           # X-Request-ID injection
│   ├── routes/
│   │   └── email_routes.py         # POST /email/process, /email/store, GET /health
│   ├── services/
│   │   ├── email_service.py        # Full pipeline orchestrator
│   │   ├── cache_service.py        # In-memory + Redis cache
│   │   └── confidence_service.py   # Multi-signal fusion
│   ├── models/
│   │   ├── email_model.py          # Email ORM (with thread_id, prompt_version, etc.)
│   │   └── thread_model.py         # ThreadMessage ORM
│   ├── schemas/
│   │   └── email_schema.py         # Pydantic request/response schemas
│   └── database/
│       └── db.py                   # SQLAlchemy engine + session factory
│
├── rag/
│   ├── embedder.py                 # sentence-transformers embedding (cached)
│   ├── vector_store.py             # FAISS index with metadata registry
│   ├── retriever.py                # Dual retrieval: FAISS + thread history
│   ├── generator.py                # LLM generation with retry + self-eval
│   └── prompts/
│       ├── v1.txt                  # Basic prompt template
│       └── v2.txt                  # Thread-aware prompt template
│
├── classifiers/
│   └── classifier.py               # Rule-based + LLM zero-shot (combined)
│
├── ingestion/
│   └── email_cleaner.py            # HTML strip, signature removal, normalize
│
├── config/
│   └── settings.py                 # Pydantic-settings (all config fields)
│
├── data/
│   └── seed_emails.json            # 20 sample emails across 4 categories
│
├── scripts/
│   ├── seed_data.py                # Seed FAISS index from JSON
│   ├── test_pipeline.py            # CLI test runner
│   └── evaluate.py                 # Batch evaluation script
│
├── frontend/
│   └── index.html                  # Standalone HTML+JS UI
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
cd "rag based email"

# Copy and fill in your environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (or HuggingFace token)
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Seed the Knowledge Base

```bash
python scripts/seed_data.py
# Output: 20 seed emails embedded and stored in FAISS
```

### 5. Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Open the UI

Navigate to **http://localhost:8000/ui/index.html** for the browser dashboard.

API docs: **http://localhost:8000/docs**

---

## 🐳 Docker Deployment

```bash
# Start with Docker Compose (includes optional Redis)
docker-compose up --build

# With Redis caching enabled
CACHE_BACKEND=redis docker-compose up --build
```

---

## 🌐 API Reference

### `POST /email/process`

Process an email through the full pipeline.

**Request:**
```json
{
  "sender": "customer@example.com",
  "subject": "My order arrived damaged",
  "body": "I received order #45821 today and the product is broken. I want a refund.",
  "thread_id": "thread-abc123"
}
```

**Response:**
```json
{
  "request_id": "f3a2b1c4-...",
  "category": "Complaint",
  "response": "We sincerely apologize for the damaged item you received...",
  "confidence": 0.812,
  "needs_review": false,
  "confidence_breakdown": {
    "classification": 0.87,
    "similarity": 0.79,
    "llm_self": 0.83,
    "final": 0.812
  },
  "retrieval_count": 5,
  "thread_id": "thread-abc123",
  "prompt_version": "v2",
  "latency_ms": 1842.3
}
```

---

### `POST /email/store`

Seed the knowledge base with a known email.

```json
{
  "sender": "user@example.com",
  "subject": "Product question",
  "body": "Do you offer a student discount?",
  "category": "Inquiry"
}
```

---

### `GET /health`

```json
{
  "status": "ok",
  "version": "2.0.0",
  "vector_store": "faiss",
  "vector_count": 20,
  "database": "ok",
  "llm_provider": "openai",
  "cache_backend": "memory",
  "prompt_version": "v2"
}
```

---

## 🧪 Sample cURL Requests

```bash
# Process an email
curl -X POST http://localhost:8000/email/process \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "john@example.com",
    "subject": "Cannot login",
    "body": "I cannot log into my account. My password reset is not working either."
  }'

# Thread-aware follow-up
curl -X POST http://localhost:8000/email/process \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "john@example.com",
    "subject": "Re: Cannot login",
    "body": "Still having the same issue after trying your suggestion.",
    "thread_id": "thread-john-001"
  }'

# Store a seed email
curl -X POST http://localhost:8000/email/store \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "seed@example.com",
    "body": "My product is damaged and I want a refund.",
    "category": "Complaint"
  }'

# Health check
curl http://localhost:8000/health
```

---

## 🔧 CLI Scripts

```bash
# Seed FAISS index
python scripts/seed_data.py
python scripts/seed_data.py --reset --verbose    # Reset and reseed

# Test pipeline locally
python scripts/test_pipeline.py
python scripts/test_pipeline.py --body "Can't login" --no-llm
python scripts/test_pipeline.py --thread-id thread-001 --body "Follow up"

# Batch evaluation
python scripts/evaluate.py
python scripts/evaluate.py --no-llm --output reports/eval.json
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `huggingface` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `VECTOR_STORE` | `faiss` | `faiss` or `chromadb` |
| `TOP_K_RETRIEVAL` | `5` | Similar emails to retrieve |
| `THREAD_HISTORY_LIMIT` | `10` | Max thread messages in prompt |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this → human review |
| `ENABLE_LLM_SELF_EVAL` | `true` | LLM rates its own response |
| `PROMPT_VERSION` | `v2` | `v1` (basic) or `v2` (thread-aware) |
| `LLM_MAX_RETRIES` | `3` | Retry attempts on API failures |
| `LLM_RETRY_BASE_DELAY` | `1.0` | Base delay for exponential backoff |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |

---

## 🧠 Confidence Scoring

Final confidence is a weighted fusion of three independent signals:

```
final = (0.4 × classification_confidence)
      + (0.4 × avg_similarity_score)
      + (0.2 × llm_self_score)
```

| Signal | Weight | Source |
|---|---|---|
| Classification | 40% | Combined rule-based + LLM zero-shot |
| Retrieval similarity | 40% | Average cosine similarity from FAISS |
| LLM self-evaluation | 20% | LLM rates quality of its own response |

If `final < confidence_threshold` → `needs_review: true`

---

## 🔮 Future Improvements

- [ ] PostgreSQL support for production-scale persistence
- [ ] Email ingestion via IMAP/SMTP webhooks
- [ ] Active learning loop: human corrections improve the model
- [ ] A/B testing between prompt versions
- [ ] Streaming response support (Server-Sent Events)
- [ ] Multi-language support via multilingual embeddings
- [ ] Slack/Teams notification for emails needing human review
- [ ] Prometheus + Grafana metrics dashboard
- [ ] Fine-tuned classification model as alternative to zero-shot

---

## 📄 License

MIT License — built for educational and production use.
#   R A G - E m a i l - A u t o m a t i o n - S y s t e m  
 #   R A G - E m a i l - A u t o m a t i o n - S y s t e m  
 