# AKSIS UI Prototype - Phase 1

This is the standalone prototype interface for the proprietary AKSIS Machine Learning framework.
It demonstrates the architecture and provides a stable API contract (via FastAPI) to decouple the UI (Streamlit) from the underlying ML backend execution.

## Project Architecture

1. **Frontend (Streamlit):** Located in `frontend/`. Responsible ONLY for presentation. Communicates with the backend exclusively via HTTP.
2. **Backend (FastAPI):** Located in `backend/`. Exposes the stable REST API (`/api/v1`).
3. **Services:**
   - **Mock Mode (`MockAksisService`):** Uses an in-memory store and fixtures to provide deterministic demo data for local UI development.
   - **AKSIS Mode (`RealAksisService`):** An adapter layer. Contains placeholders that translate API schemas into actual AKSIS `ExperimentConfig` and calls the real AKSIS pipeline.

## Getting Started

### 1. Environment Setup

Create your `.env` file:
```bash
cp .env.example .env
```
Ensure `AKSIS_PROVIDER=mock` is set for local development without the real framework.

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start FastAPI Backend

```bash
uvicorn backend.main:app --reload
```
View the OpenAPI Docs (Swagger) at: http://127.0.0.1:8000/docs

### 3. Start Streamlit Frontend

In a separate terminal:
```bash
cd frontend
streamlit run app.py
```
View the UI at: http://localhost:8501

## Integration Notes

For details on connecting `RealAksisService` to the actual AKSIS framework on the company computer, see [INTEGRATION.md](INTEGRATION.md).
