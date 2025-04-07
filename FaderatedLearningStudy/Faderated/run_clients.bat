@echo off
for /L %%i in (1,1,8) do (
    start cmd /k python client.py
)
