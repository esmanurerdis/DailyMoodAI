@echo off
python -m scripts.main ui --port 7860
start http://127.0.0.1:7860
pause
