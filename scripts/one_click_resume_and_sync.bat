@echo off
setlocal
cd /d "%~dp0\.."
python -u scripts\tmp_paramiko_autodl_runner.py --auto-resume
endlocal
