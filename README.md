# Agentic Marketing System

AI-powered marketing campaign analysis backend. Built with FastAPI, GPT-4o, and Instructor for structured outputs.

## What it does

Analyzes marketing campaign performance and returns structured, confidence-scored recommendations. Input ad spend, revenue, product price, and traffic source — get back a performance score, key findings, prioritized action items, and reasoning.

## Architecture

```
React + Vite Frontend (Vercel)
        ↓
FastAPI Backend (Railway)  ← this repo
        ↓              ↓
   OpenAI GPT-4o    PostgreSQL
   (Instructor)     (Supabase)
        ↓
   LangSmith (tracing)
```

## Tech Stack

- **Framework:** FastAPI (Python)
- **AI:** OpenAI GPT-4o via Instructor library (structured outputs)
- **Database:** PostgreSQL via Supabase (Session Pooler)
- **Deployment:** Railway
- **Observability:** LangSmith
- **Validation:** Pydantic schemas

## Project Structure

```
├── routers/        # HTTP route handlers
├── services/       # Business logic and AI orchestration
├── schemas/        # Pydantic models for request/response validation
├── db/             # Database connection and queries
├── main.py         # FastAPI app entry point
├── models.py       # SQLModel database models
├── database.py     # Database initialization
└── requirements.txt
```

## Key Design Decisions

**Structured outputs via Instructor:** Every AI response is validated against a Pydantic schema before reaching the frontend. No free-form text parsing — typed, reliable output every time.

**Tiered required fields:** Only `spend` and `revenue` are required. All other metrics are auto-calculated or flagged as missing with confidence scoring. This avoids penalizing data-sparse users.

**Session Pooler for Railway:** Railway operates on IPv4; Supabase direct connections require IPv6. The Session Pooler connection string resolves this — use `postgresql://postgres.[ref]:[password]@aws-0-us-west-2.pooler.supabase.com:5432/postgres` format.

## Local Setup

```bash
# Clone the repo
git clone https://github.com/joannpraise-oss/agentic-marketing-system
cd agentic-marketing-system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your actual values (see Environment Variables below)

# Run the development server
uvicorn main:app --reload
```

## Environment Variables

Create a `.env` file in the root directory (never commit this file):

```
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_supabase_session_pooler_connection_string
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
```

## API Endpoints

### POST `/analyze`
Analyzes a marketing campaign and returns structured recommendations.

**Request body:**
```json
{
  "spend": 150.00,
  "revenue": 380.00,
  "product_price": 32.00,
  "traffic_source": "Meta Ads"
}
```

**Response:**
```json
{
  "performance_score": 7,
  "overall_assessment": "Campaign is profitable but below optimal ROAS...",
  "key_findings": ["ROAS of 2.53 exceeds break-even", "..."],
  "action_items": ["Test higher-intent audiences", "..."],
  "confidence_level": "high",
  "reasoning": "..."
}
```

### GET `/health`
Health check endpoint.

## Deployment

The backend is deployed on Railway. The `Procfile` configures the start command:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set all environment variables in Railway's Variables tab. Never hardcode credentials.

## Frontend

The React + Vite frontend is in a separate repository:
- **Repo:** https://github.com/joannpraise-oss/agentic-marketing-ui
- **Live:** https://agentic-marketing-ui.vercel.app

## Security Notes

- All API keys and credentials are managed via environment variables
- `.gitignore` excludes `.env` files from version control
- No credentials are committed to this repository
- Database password rotation is implemented as a security practice
