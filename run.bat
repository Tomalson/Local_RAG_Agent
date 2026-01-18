@echo off
REM Advanced RAG System Launcher
REM Aktywuje venv i uruchamia main.py

cd /d "%~dp0"
call ..\..venv\Scripts\activate.bat
python main.py
pause
