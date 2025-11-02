@echo off
title Neural Trading Cyberpunk Launcher
color 0A

echo.
echo ===============================================
echo    NEURAL TRADING CYBERPUNK LAUNCHER
echo ===============================================
echo.
echo 🤖 Sistema de Trading com IA Neural
echo ⚡ GPU Acceleration Ready  
echo 🎯 Estratégias Avançadas
echo.
echo ===============================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado!
    echo Por favor, instale Python 3.7+ primeiro.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python encontrado!
echo.

REM Executar o launcher Python
echo 🚀 Iniciando Neural Trading...
cd /d "%~dp0"
python start_neural_cyberpunk.py

if errorlevel 1 (
    echo.
    echo ❌ Erro ao executar o sistema!
    pause
    exit /b 1
)

echo.
echo 👋 Obrigado por usar Neural Trading!
pause
