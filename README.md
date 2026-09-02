# AI Ticket Resolution System

> For a detailed breakdown of the architecture, data flow, models, API, and design decisions, see [`SRS.md`](./SRS.md) . This README is just a quick overview.

A prototype that reads an incoming support ticket, pulls similar past tickets from a knowledge base, drafts an answer with an LLM, and either sends it automatically or hands it to an admin for review — depending on how confident the system is.

## The problem

Support teams spend a lot of time on repetitive work: reading a ticket, searching old tickets or docs for the answer, writing a reply, and sending it. Most of that follows the same pattern every time, even though every ticket looks different on the surface.

## The solution

This project automates that loop: retrieve relevant past tickets, draft an answer with an LLM, and route it — either straight to the customer, or to an admin for review, depending on how confident the system is in the draft. It doesn't resolve every ticket on its own; it only auto-sends when several independent checks agree the answer is safe, and everything else goes to an admin queue.

## How it works

```
Ticket in → Retrieve similar past tickets (Chroma) → LLM drafts an answer
          → Grounding check → Decision gate → auto-send OR admin review
```

1. A ticket comes in through the web form or the API.
2. It's embedded and used to find similar past tickets in a vector store.
3. An LLM drafts a reply using that context, along with a confidence score and any risk flags.
4. A second, independent LLM call checks how well the draft is actually backed by the retrieved context.
5. If retrieval strength, confidence, and grounding all check out — and no risk flags were raised — the answer is sent to the customer and saved back into the knowledge base.
6. Otherwise it goes to an admin dashboard for review, editing, and approval before it goes out.

## Tech stack

- **Backend:** FastAPI (Python)
- **Vector store:** ChromaDB
- **Embeddings:** BGE-large-en-v1.5 (local, via `sentence-transformers`)
- **LLM:** OpenAI-compatible chat API (currently Groq, `openai/gpt-oss-20b`)
- **Database:** SQLite
- **Frontend:** plain HTML/CSS/JS, served by the same FastAPI app
- **Containerization:** Docker

## Getting started

```bash
git clone <repo-url>
cd ticket_rag_system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# create a .env file with:
#   GROQ_API_KEY=...
#   ADMIN_PASSWORD=...
#   AUTH_TOKEN_SECRET=...
#   SMTP_PASSWORD=...        (only if using email notifications)

python scripts/build_chroma_index.py     # builds the vector index locally
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` for the UI, or `/docs` for the interactive API docs.

To try it without an API key, set `USE_MOCK_LLM=true` and run the same commands — it uses offline mock stand-ins instead of a real model.

Run the test suite:

```bash
python tests/test_pipeline.py
```

Or with Docker:

```bash
docker build -t ticket-rag-system .
docker run -p 8000:8000 --env-file .env ticket-rag-system
```

## Status

Working prototype — containerized with Docker, but not deployed to any cloud environment yet. See `SRS-Doc.pdf` for what's implemented vs. tested vs. still planned.

## Future updates

- A real user/admin account system, replacing the current shared-password gate
- A feedback loop that uses admin edits to improve future drafts
- Automated evaluation of answer quality
- Monitoring and observability
- Production cloud deployment

## Author

Built by [Umer Farooq](https://github.com/umer-farooq230).
