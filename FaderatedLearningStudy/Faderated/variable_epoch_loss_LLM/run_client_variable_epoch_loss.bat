@echo off
for /L %%i in (1,1,1) do (
    start cmd /k python client_variable_epoch_loss_LLM.py
)
