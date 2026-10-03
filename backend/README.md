# Accident Detection — Backend

FastAPI inference server that powers the Next.js frontend.

## Setup

```bash
# From repo root, install deps (ML libs already installed from Streamlit app)
pip3 install fastapi "uvicorn[standard]" python-multipart

# Start the server
cd backend
uvicorn main:app --reload --port 8000
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model load status + model stats |
| `/detect/image` | POST | Image inference → annotated base64 + boxes |
| `/detect/video` | POST | Video inference → SSE progress stream |
| `/detect/video/download/{job_id}` | GET | Download processed video |

## Notes
- Video processing is CPU-bound. For a 30-second clip at ~600ms/frame you'll wait ~18 seconds.
- Processed videos are stored in `/tmp/accident_detection_jobs/` and are NOT auto-deleted (clean up manually or restart).
- CORS is configured for `http://localhost:3000`.
