@echo off


call activate ExperimentTwo

for /l %%i in (1, 1, 20) do (

    cd /d "%~dp0"

    python Agent.py %%i

    timeout /t 5 /nobreak > nul
)
