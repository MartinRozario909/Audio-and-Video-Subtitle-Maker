@echo off
cd /d "%~dp0"
set "PY=E:\Python 3.11.9\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" whisper_srt_qt.py
if errorlevel 1 pause
