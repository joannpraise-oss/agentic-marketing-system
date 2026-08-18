# Agentic Marketing Optimization System

**Brkeven** — AI-powered campaign analysis for solo founders and small marketing agencies.

Live at: [brkeven.com](https://brkeven.com) · [API Docs](https://agentic-marketing-system-production.up.railway.app/docs)

---

## What It Does

Brkeven analyzes your ad campaign performance using AI and returns instant, actionable insights — no spreadsheets, no guesswork.

You provide your ad spend and revenue. The system diagnoses what's working, identifies the root cause of underperformance, suggests specific optimizations, and scores its own confidence based on the data available.

---

## Live Demo

**Frontend:** https://brkeven.com  
**Backend API:** https://agentic-marketing-system-production.up.railway.app/docs

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI (Python), deployed on Railway |
| Database | PostgreSQL via Supabase |
| AI | OpenAI GPT-4o via Instructor (structured outputs) |
| Observability | LangSmith tracing |

---

## Architecture

```
User (browser)
    ↓
React Frontend — Vercel (brkeven.com)
    ↓ POST /analyze_campaign
FastAPI Backend — Railway
    ↓                        ↓
OpenAI GPT-4o          Supabase PostgreSQL
(campaign analysis)    (campaign + result storage)
    ↓
LangSmith
(trace logging)
```

---

## API

### POST `/analyze_campaign`

Analyzes a campaign and returns an AI-generated diagnosis.

**Request body:**
```json
{
  "spend": 500.00,
  "revenue": 2000.00,
  "product_price": 32.00,
  "traffic_source": "Meta Ads"
}
```

`spend` and `revenue` are required. All other fields are optional — the system applies confidence scoring when data is missing.

**Response:**
```json
{
  "campaign_id": 21,
  "performance_diagnosis": "Strong ROAS of 4.0x indicates healthy campaign performance...",
  "root_cause": "High revenue relative to spend suggests effective targeting...",
  "optimization_suggestion": "Consider scaling budget by 20-30% while monitoring ROAS...",
  "confidence_score": 0.82,
  "metrics_used": ["spend", "revenue"],
  "missing_metrics": ["ctr", "cac", "impressions"],
  "tokens_used": 312,
  "estimated_cost_usd": 0.0031
}
```

---

## Key Design Decisions

**Tiered data intelligence:** Only `spend` and `revenue` are required. The AI adjusts its confidence score based on which optional metrics are provided — more data yields higher confidence, but the system never refuses to analyze due to missing fields.

**Structured AI outputs:** GPT-4o responses are validated against a Pydantic schema via Instructor, ensuring the API always returns consistent, well-typed data rather than freeform text.

**Session Pooler for Railway + Supabase:** Railway's network cannot reach Supabase's direct IPv6 connection. The Supabase Session Pooler (IPv4-compatible) is required for Railway compatibility.

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- A Supabase project (PostgreSQL)
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Backend

```bash
git clone https://github.com/joannpraise-oss/agentic-marketing-system
cd agentic-marketing-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
DATABASE_URL=your_supabase_session_pooler_uri
OPENAI_API_KEY=your_openai_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=agentic-marketing
```

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

### Frontend

```bash
git clone https://github.com/joannpraise-oss/agentic-marketing-ui
cd agentic-marketing-ui
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## Project Roadmap

This is Phase 1 of a 5-phase system:

| Phase | Scope | Status |
|---|---|---|
| 1 | Ad Performance Analyzer | ✅ Live |
| 2 | Agentic orchestration + Advisory Mode | 🔜 Next |
| 3 | Multi-tenant billing + auth | Planned |
| 4 | Analytics and pattern detection (product moat) | Planned |
| 5 | Creative Orchestration Layer + Campaign Launcher | Specced — [view spec](docs/agentic-campaign-launcher-spec.md) |

**Phase 5 vision:** A solo founder types *"I need a campaign for postpartum moms"* and the system generates ad creatives, sets up targeting, and launches directly to Meta — in under 15 minutes. See the [full spec](docs/agentic-campaign-launcher-spec.md).

---

## Repositories

- **Backend:** [joannpraise-oss/agentic-marketing-system](https://github.com/joannpraise-oss/agentic-marketing-system)
- **Frontend:** [joannpraise-oss/agentic-marketing-ui](https://github.com/joannpraise-oss/agentic-marketing-ui)

---

*Built by [Joann Praise Emmanson-Ogbeide](https://github.com/joannpraise-oss) · Brkeven · 2026*
