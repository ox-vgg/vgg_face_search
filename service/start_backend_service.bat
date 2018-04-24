@echo off
cd %~dp0
cmd /k "..\..\Scripts\activate & python backend.py"
