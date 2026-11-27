@echo off
cd /d "%~dp0"
echo Starting Spam Shield on http://localhost:8001
start "" http://localhost:8001
".venv\Scripts\python.exe" -m uvicorn src.api.app:app --port 8001
