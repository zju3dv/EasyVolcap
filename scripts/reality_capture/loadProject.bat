@echo off

:: Start a new RealityCapture instance.
start "" /b %RCexe% -setInstanceName %instanceName% -silent %metadata%

echo Starting a new RealityCapture instance.

:: Define the wait start label.
:waitStart
:: Check the status of the RealityCapture instance and wait for the activation to finish.
%RCexe% -getStatus %instanceName%
IF /I "%ERRORLEVEL%" NEQ "0" (
    echo Waiting for the activation.
    goto :waitStart
)

echo Activation finished.

:: Change the error settings in RC and load the project. 
%RCexe% -delegateTo %instanceName% -set "appQuitOnError=true" -waitCompleted %instanceName%
%RCexe% -delegateTo %instanceName% -load %projectFile% -waitCompleted %instanceName%