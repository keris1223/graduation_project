@echo off
for /L %%i in (1,1,2) do (
    start cmd /k python fixed_epoch_client_LLM.py
)
