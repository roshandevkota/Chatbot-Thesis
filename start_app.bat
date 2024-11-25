@echo off
REM Batch script to activate virtual environment and start Flask app

REM Navigate to project directory
cd /d D:\Thesis\Chatbot-Thesis
if %ERRORLEVEL% neq 0 (
    echo Failed to navigate to project directory.
    pause
    exit /b %ERRORLEVEL%
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Failed to activate the virtual environment.
    pause
    exit /b %ERRORLEVEL%
)

REM Start Flask application
python src\app.py
if %ERRORLEVEL% neq 0 (
    echo Flask application terminated with errors.
    pause
    exit /b %ERRORLEVEL%
)

REM Keep the window open after the app stops
pause
