@echo off
set app="C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe"

rem Test the app is running

%app% -getStatus *
IF /I "%ERRORLEVEL%"=="0" (
    
    echo RealityCapture istance is already running
    %app% -delegateTo RC1 -newScene
    goto :eof
)

echo Starting new RealityCapture istance

start "" %app% -headless -setInstanceName RC1

echo Waiting until the app starts

:waitStart
%app% -getStatus *
IF /I "%ERRORLEVEL%" NEQ "0" (
    goto :waitStart
)

:eof