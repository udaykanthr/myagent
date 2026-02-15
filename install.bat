@echo off
echo Installing multi-agent coder...

pip install -e .

if %ERRORLEVEL% EQU 0 (
    echo Installation successful! You can now run 'agentchanti' from anywhere.
) else (
    echo Installation failed.
)
pause
