@echo off
set app="C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe"

rem Waits for RealityCapture to be done with the process so the batch file can continue
%app% -waitCompleted RC1

