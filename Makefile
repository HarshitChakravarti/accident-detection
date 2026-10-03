.PHONY: backend frontend dev

# Start FastAPI backend
backend:
	cd backend && uvicorn main:app --reload --port 8000

# Start Next.js frontend dev server
frontend:
	cd frontend && npm run dev

# Start both concurrently (requires GNU make -j or parallel shell)
dev:
	@echo "Starting backend on :8000 and frontend on :3000…"
	@(make backend &) && make frontend

# Build frontend for production
build:
	cd frontend && npm run build

# Lint frontend
lint:
	cd frontend && npm run lint
