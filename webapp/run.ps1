# DataMind AI - Web Application

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DataMind AI - Database Documentation" -ForegroundColor White
Write-Host "  No API Key Required!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "Starting DataMind AI Web Server..." -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser and go to:" -ForegroundColor White
Write-Host "  http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

python app.py
