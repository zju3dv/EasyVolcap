@echo off

:: Call the batch file with variables.
call setVariables.bat

:: Create a variable for the number of lines in the command file.
set lines=0

:: Calculate the number of lines in the file with commands.
for /F "tokens=*" %%A in (%commandFile%) do (set /a lines+=1)

:: A variable used to limit the number of allowed crash occurences.
set /a crashNumber=0

:: A variable used to limit the number of allowed error occurences.
set /a errorNumber=0

:: The crash label. After a crash, the process will start from here.
:crashCase

:: Call the batch file that starts a new instance.
call startApp.bat

:: Enable delayed expansion for the variables that will change during the FOR cycle.
setlocal enabledelayedexpansion

:: Set a variable that is going to be used to determine which line of the command file is being used.
set lineNumber=1

:: The label that starts the RealityCapture process
:process

:: Set a counter variable used for the FOR cycle to determine which line is going to be run.
set /a counter=0

:: Start the cycle process through the command file lines.
for /F "tokens=*" %%B in (%commandFile%) do (
    
    :: Increase the counter value by one each cycle.
    set /a counter+=1
    
    :: If the counter value equals the lineNumber value, then run a line from the command line that corresponds to the lineNumber value.
    if !counter! EQU %lineNumber% (
        echo "!time! Start: command line #%lineNumber%: %%B" >> %rootFolder%\metadata\processLog.txt
        %RCexe% -delegateTo %instanceName% %%B
        %RCexe% -waitCompleted %instanceName%
        echo "!time! End: command line #%lineNumber%: %%B" >> %rootFolder%\metadata\processLog.txt
        %RCexe% -delegateTo %instanceName% -save %projectFile%
        %RCexe% -waitCompleted %instanceName%
    )    
    
    :: Check the RealityCapture log file for the string "Processing failed". If the string is there, proceed to the error case label.
    find /c "Processing failed:" C:\Users\%username%\AppData\Local\Temp\RealityCapture.log > NUL
    if "!ERRORLEVEL!" EQU "0" (
        echo "!time! ERROR at line #%lineNumber%: %%B" >> %rootFolder%\metadata\processLog.txt
        goto errorCase
        )  
    
    :: Write the RC status into the status.txt. If this is not possible, continue with the crash case label.
    %RCexe% -getStatus %instanceName% > %rootFolder%\metadata\status.txt
    if "!ERRORLEVEL!" NEQ "0" (
        echo "!time! CRASH at line #%lineNumber%: %%B" >> %rootFolder%\metadata\processLog.txt
        :: Increase the crash number value by one.
        set /a crashNumber+=1
        :: In case the crash happened three times already, go to the end of the file.
        if !crashNumber! EQU 3 (
            echo "Maximum number of tries exceeded." >> %rootFolder%\metadata\processLog.txt
            goto eof
            )
        echo "Process restarted at !time!" >> %rootFolder%\metadata\processLog.txt
        goto :crashCase
    )       
)
  
:: Increase the line number value by one to go to the next line in the command file.  
set /a lineNumber+=1

:: If the line number value exceedes the lines value, go to the end of the file.
if !lineNumber! GTR %lines% (goto eof)

:: Go back to the RealityCapture processing.
goto process

:: Define the error case label.
:errorCase

:: In case the error happened three times already, go to the end of the file.
if %errorNumber% EQU 3 (
    echo "Maximum number of tries exceeded." >> %rootFolder%\metadata\processLog.txt
    goto eof
)

:: Increase the error number value by one.
set /a errorNumber+=1

:: Run the loadProject.bat after the error.
call loadProject.bat

echo "Instance restarted at !time!" >> %rootFolder%\metadata\processLog.txt

:: Go back to the RealityCapture processing.
goto process

:: Define the end of the file label.
:eof

@pause









