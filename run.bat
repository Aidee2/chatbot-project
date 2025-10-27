@echo off
REM Create a virtual environment if not present
IF NOT EXIST venv (
   python -m venv venv
)
REM Activate the virtual environment
call venv\Scripts\activate
REM Install requirements automatically
pip install -r requirements.txt
REM Run your app
python app.py
pause
