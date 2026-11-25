@echo off
REM Starts the Feedback Lens app (FastAPI backend + frontend) on http://127.0.0.1:8000
cd /d "%~dp0"
echo Starting Feedback Lens at http://127.0.0.1:8000  (API docs: /docs)
".venv\Scripts\python.exe" -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
