@echo off
title AI ClipMaker F1

echo.
echo  ================================================
echo    AI ClipMaker F1 - Starting up...
echo  ================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (python3 --version >nul 2>&1
    if errorlevel 1 (echo  [!] Python not installed. Go to https://www.python.org/downloads && pause && exit /b 1)
    set PYTHON=python3 && set PIP=pip3) else (set PYTHON=python && set PIP=pip)

echo  [OK] Python found.
echo  [..] Checking required packages...
echo.
for %%P in (streamlit moviepy pandas requests plotly anthropic gtts) do (
    %PYTHON% -m pip show %%P >nul 2>&1
    if errorlevel 1 (echo  [..] Installing %%P ... && %PIP% install %%P --quiet))

echo.
echo  [i] Optional AI packages (install manually if needed):
echo      pip install librosa scipy          (Phase 2 - Audio AI)
echo      pip install ultralytics opencv-python  (Phase 3 - Vision AI)
echo.

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo  [!] FFmpeg not found. Download: https://ffmpeg.org/download.html
    echo      Extract and add the bin folder to your PATH.
    echo.)

echo  [OK] Core packages ready.
echo.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml")

echo  [..] Opening AI ClipMaker F1 in your browser...
echo  Keep this window open. Close it when done.
echo  ================================================
echo.
%PYTHON% -m streamlit run "%~dp0app_streamlit.py" --server.headless false --browser.gatherUsageStats false
pause
