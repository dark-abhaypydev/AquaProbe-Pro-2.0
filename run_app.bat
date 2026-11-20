@echo off
title AquaProbe Pro-9™ - Physicochemical Water Analyzer
cd /d "%~dp0"
echo ========================================================
echo Launching AquaProbe Pro-9™ Sensor Console
echo Model: AP-9000 PRO | Environmental Diagnostics Suite
echo ========================================================
python -m streamlit run app.py
pause
