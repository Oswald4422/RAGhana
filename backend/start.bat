@echo off
:: Kill anything on port 8001 first
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTEN') do (
    taskkill /F /PID %%a >nul 2>&1
)
:: Start the API
python -m uvicorn api:app --port 8001 --reload
