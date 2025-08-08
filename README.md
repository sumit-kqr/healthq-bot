# HealthQ

A lightweight RAG chatbot to answer queries about health insurance policies.

## Features
- Conversational RAG flow with chat history
- PDF ingestion and chunking
- Vector search via Chroma
- OpenAI LLM + embeddings (no heavyweight local models)
- Streamlit UI and optional FastAPI API

## Requirements
- Python 3.10+ (Docker image uses 3.10-slim)
- OpenAI API key
- (Optional) HF token for future HF use

## Environment Variables
- `OPENAI_API_KEY` (required)
- `HF_TOKEN` (optional)

If using a `.env` file, place it in the project root:
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

## Local Development (without Docker)
1. Create a virtualenv and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   App will be available at `http://localhost:8501`.

3. Optional: Run the FastAPI server (API only):
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   Health check: `GET /ping` → `{ "status": "ok" }`.

## Docker
Build and run the Streamlit app:
```bash
docker build -t healthq:latest .
docker run -d --name healthq_app -p 8501:8501 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  healthq:latest
```

Notes:
- The image is based on `python:3.10-slim`.
- `.dockerignore` excludes heavy local artifacts (like `chroma_db/`) from the build context.

## Deploy to Render (Blueprint)
This repo includes `render.yaml` to deploy using Render’s Blueprint.

Steps:
1. Push the repo to GitHub (this repo is prepared for it).
2. Go to `https://render.com` → New → Blueprint.
3. Select your GitHub repo and approve access.
4. Render will detect `render.yaml` and create a Docker-based Web Service.
5. Set environment variables in the service:
   - `OPENAI_API_KEY` (required)
   - `HF_TOKEN` (optional)
6. Create and deploy. Render will build using the included `Dockerfile`.

The `Dockerfile` command runs Streamlit and binds to the port provided by Render via `$PORT`:
```
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
```

## Optional: API-only deploy (lighter)
If you want a lighter deploy that exposes the FastAPI server only:
- Remove `streamlit` from `requirements.txt`.
- Change the Dockerfile CMD to:
  ```
  CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
  ```
This reduces size by removing Streamlit’s heavier UI dependencies.

## What Changed to Reduce Image Size
- Switched embeddings to OpenAI:
  - Replaced `HuggingFaceEmbeddings` with `OpenAIEmbeddings` in `modules/vector_store.py`.
  - Avoids downloading PyTorch and HF model artifacts at build/runtime.
- Trimmed dependencies in `requirements.txt`:
  - Removed `sentence_transformers` and `langchain_huggingface`.
  - Kept `langchain_chroma`, `chromadb`, and other essentials only.
- Added `.dockerignore`:
  - Excludes `chroma_db/`, caches, PDFs, venvs, etc., so they are not copied into the image.
- Dockerfile improvements:
  - Stay on `python:3.10-slim` and run Streamlit via `$PORT` (Render-compatible).
- Stability fix:
  - Guarded `HF_TOKEN` usage in `app.py` so it’s optional (prevents `NoneType` crash).

Result: Image size reduced to ~1.28GB from multiple GB and avoids runtime crashes when `HF_TOKEN` is unset.

## Troubleshooting
- Streamlit shows but errors on PDF load:
  - Ensure the uploaded file is a valid PDF.
- 401/403 errors from OpenAI:
  - Check `OPENAI_API_KEY` in environment variables.
- GPU/Model download timeouts:
  - None expected now since embeddings use OpenAI rather than local HF models.
- Large repo or slow Docker build:
  - Confirm `.dockerignore` and `.gitignore` exclude `chroma_db/`, caches, and artifacts.

## License
MIT 
