# AI Agent Evaluation Pipeline

A pipeline that automatically evaluates AI agent conversations, integrates human feedback, and generates improvement suggestions — designed for 1000+ conversations/minute.

---

## Architecture

```
Client
  │
  ▼
FastAPI (controllers)          ← HTTP boundary, returns 202 immediately
  │
  ▼
Services                       ← Business logic, validation
  │
  ▼
Kafka Producer ──────────────► Kafka Topics
                                  ├── conversations.raw
                                  ├── feedback.raw
                                  └── evaluations.trigger
                                         │
                                         ▼
                               Kafka Workers (async consumers)
                                  ├── ConversationWorker
                                  ├── FeedbackWorker
                                  ├── EvaluationWorker  ──► 4 independent evaluators
                                  │                          ├── LLM-as-Judge (Claude)
                                  │                          ├── Tool Call Evaluator
                                  │                          ├── Coherence Evaluator
                                  │                          └── Heuristic Evaluator
                                  └── ProcessedFeedbackWorker
                                         │
                                         ▼
                                     MongoDB
                                  ├── conversations
                                  ├── feedback
                                  ├── evaluations
                                  ├── suggestions
                                  └── meta_evaluations
```

**Key design decisions:**
- The API never blocks — it publishes to Kafka and returns `202 Accepted`. Workers process asynchronously.
- The 4 evaluators run concurrently via `asyncio.gather()` and are independently scalable consumer groups.
- All I/O is async (Motor for MongoDB, aiokafka for Kafka).
- Kafka decouples ingestion rate from processing rate — horizontal scaling is achieved by adding partitions and consumer instances.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115 + uvicorn |
| DB | MongoDB 7.0 via Motor 3.6 (async) |
| Queue | Apache Kafka 3.7 (KRaft, no Zookeeper) |
| Validation | Pydantic v2 + pydantic-settings |
| LLM Judge | Anthropic SDK (Claude) |
| Testing | pytest + pytest-asyncio + httpx |
| UI | Streamlit |

---

## Running Locally

### Prerequisites
- Docker + Docker Compose
- Python 3.12+

### 1. Clone and configure

```bash
git clone <repo-url>
cd AI_Agent_Evaluation_Pipeline
cp .env.example .env
```

Edit `.env` — at minimum set your `ANTHROPIC_API_KEY` and verify the Mongo credentials.

### 2. Start infrastructure + app

```bash
docker compose up --build
```

This starts MongoDB, Kafka, and the FastAPI app. The API will be available at `http://localhost:8000`.

### 3. Verify

```bash
curl http://localhost:8000/health
```

API docs: `http://localhost:8000/docs`

---

## Running Tests

```bash
pip install -r requirements.txt
pytest
```

---

## Running the Streamlit Dashboard

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

---

## Project Structure

```
src/
  config/        # Pydantic settings (all config from .env, no hardcoded secrets)
  controllers/   # FastAPI routers — ingestion, evaluation, feedback, suggestions
  services/      # Business logic, Kafka producer, feedback aggregation, suggestion generation
  data/
    schemas/     # Pydantic request/response models
    repositories/ # MongoDB CRUD (Motor async)
    database.py  # Connection lifecycle
  evaluators/    # 4 independent evaluators (LLM judge, tool call, coherence, heuristic)
  workers/       # Kafka consumers — persist and trigger evaluation
  utils/         # Exceptions, FastAPI dependency injection
tests/           # Full test coverage across all layers
scripts/         # mongo-init.js — creates collections, validation schemas, indexes
docs/            # Architecture deep-dive, file map, test coverage notes
```

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ingest/conversation` | Ingest a conversation log |
| `POST` | `/api/v1/ingest/batch` | Batch ingest conversations |
| `POST` | `/api/v1/feedback` | Submit human feedback |
| `POST` | `/api/v1/evaluations/trigger` | Trigger evaluation for a conversation |
| `GET` | `/api/v1/evaluations/{conversation_id}` | Fetch evaluation results |
| `GET` | `/api/v1/suggestions` | List improvement suggestions |
| `GET` | `/api/v1/meta-evaluations` | Evaluator quality metrics |

---

## Environment Variables

See `.env.example` for the full list. Key variables:

```
MONGO_URI=mongodb://evaladmin:evalpass123@localhost:27017/eval_pipeline?authSource=admin
KAFKA_BOOTSTRAP_SERVERS=localhost:9094   # outside Docker; kafka:9092 inside Docker
ANTHROPIC_API_KEY=sk-...
```
