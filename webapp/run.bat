@echo off
echo ============================================
echo   DataMind AI - Database Documentation
echo   No API Key Required!
echo ============================================
echo.

cd /d "%~dp0"

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting DataMind AI Web Server...
echo.
echo Open your browser and go to:
echo   http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

python app.py
