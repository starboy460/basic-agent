@echo off
setlocal

cd /d "%~dp0"
title Neo Web

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if errorlevel 1 (
    echo Starting local Ollama server...
    start "Ollama Local Server" /min ollama serve
    timeout /t 3 /nobreak >NUL
)

echo Starting Neo web app...
echo Open: http://127.0.0.1:8765
echo Local memory: %CD%\chat_history.json
echo.
".venv\Scripts\python.exe" main.py web

echo.
echo Web app closed.
pause
