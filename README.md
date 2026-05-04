# RAG-Based Email Automation System 

A production-grade AI backend that classifies incoming emails and generates context-aware responses using Retrieval-Augmented Generation (RAG).

Built with FastAPI, LangChain, FAISS, sentence-transformers, SQLite, and Docker.

---

## Architecture


Incoming Email
│
▼
Middleware (Request ID)
│
▼
EmailService (async orchestrator)
│
├── EmailCleaner
│ └── HTML strip, signature removal, normalization
│
├── Cache Check
│ └── SHA-256 hash lookup (memory/Redis)
│
├── Classifier
│ └── Rule-based + LLM zero-shot
│
├── Retriever
│ ├── FAISS search (semantic similarity)
│ └── Thread history (previous messages)
│
├── Generator
│ └── Prompt + retry + self-evaluation
│
├── Confidence Fusion
│ └── Weighted scoring
│
└── DB Persist + Cache
└── SQLite + FAISS


---

## Project Structure


rag based email/
├── app/
│ ├── main.py
│ ├── core/
│ │ ├── logger.py
│ │ └── middleware.py
│ ├── routes/
│ │ └── email_routes.py
│ ├── services/
│ │ ├── email_service.py
│ │ ├── cache_service.py
│ │ └── confidence_service.py
│ ├── models/
│ │ ├── email_model.py
│ │ └── thread_model.py
│ ├── schemas/
│ │ └── email_schema.py
│ └── database/
│ └── db.py
│
├── rag/
│ ├── embedder.py
│ ├── vector_store.py
│ ├── retriever.py
│ ├── generator.py
│ └── prompts/
│ ├── v1.txt
│ └── v2.txt
│
├── classifiers/
│ └── classifier.py
├── ingestion/
│ └── email_cleaner.py
├── config/
│ └── settings.py
├── data/
│ └── seed_emails.json
├── scripts/
│ ├── seed_data.py
│ ├── test_pipeline.py
│ └── evaluate.py
├── frontend/
│ └── index.html
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md


---

## Quick Start

### 1. Clone and Setup

```bash
cd "rag based email"

2. Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Seed the Knowledge Base
python scripts/seed_data.py
5. Start the Server
uvicorn app.main:app --reload --port 8000
6. Open the UI
http://localhost:8000/ui/index.html

API documentation:

http://localhost:8000/docs
Docker Deployment
docker-compose up --build

With Redis caching:

CACHE_BACKEND=redis docker-compose up --build
API Reference
POST /email/process

Request:

{
  "sender": "customer@example.com",
  "subject": "My order arrived damaged",
  "body": "I received order #45821 and the product is broken.",
  "thread_id": "thread-abc123"
}

Response:

{
  "request_id": "...",
  "category": "Complaint",
  "response": "We apologize for the issue...",
  "confidence": 0.812,
  "needs_review": false,
  "confidence_breakdown": {
    "classification": 0.87,
    "similarity": 0.79,
    "llm_self": 0.83,
    "final": 0.812
  }
}
POST /email/store
{
  "sender": "user@example.com",
  "subject": "Product question",
  "body": "Do you offer a student discount?",
  "category": "Inquiry"
}
GET /health
{
  "status": "ok",
  "version": "2.0.0",
  "vector_store": "faiss"
}
Sample cURL Commands
curl -X POST http://localhost:8000/email/process \
-H "Content-Type: application/json" \
-d '{"sender":"john@example.com","body":"Cannot login"}'
CLI Scripts
python scripts/seed_data.py
python scripts/test_pipeline.py
python scripts/evaluate.py
Configuration
Variable	Description
LLM_PROVIDER	openai or huggingface
OPENAI_API_KEY	API key
VECTOR_STORE	faiss or chromadb
CACHE_BACKEND	memory or redis
PROMPT_VERSION	v1 or v2
Confidence Scoring
final = (0.4 × classification)
      + (0.4 × similarity)
      + (0.2 × llm_self)

If score is below threshold, the email is flagged for review.
