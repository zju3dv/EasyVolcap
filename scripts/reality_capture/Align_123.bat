
rem By turning the echo off, you are turning off writing every step of the process into the Command Line
@echo off

rem Setting variables for most used directories. They can be called later by putting their name into %%, e.g. %workingDir%
rem in this case workingDir is a path to the folder where this BAT file is located
set app="C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe"
set workingDir="D:\selfcap\0512_bike\colmap"

rem calls batch file which starts RealityCapture or just creates a new scene in an already created RC instance
call startApp.bat

rem add and register images from folder images1

%app% -delegateTo RC1 -addFolder "D:\selfcap\0512_bike\colmap\000203\images" -set "sfmEnableCameraPrior=true" -align -exportSparsePointCloud "D:\selfcap\0512_bike\colmap\000203\temp.ply"
rem add and register images from folder images2
%app% -delegateTo RC1 -addFolder "D:\selfcap\0512_bike\colmap\000605\images" -set "sfmEnableCameraPrior=true" -align -exportSparsePointCloud "D:\selfcap\0512_bike\colmap\000605\temp.ply"

@REM rem add and register images from folder images3
@REM %app% -delegateTo RC1 -addFolder "%workingDir%\images3" -importLicense "%workingDir%\license\ppi-licenses.rclicense" -align

@REM rem Following are RealityCapture commands that export XMP metadata files for the largest component in the scene
@REM %app% ^
@REM -delegateTo RC1 ^
@REM -selectMaximalComponent ^
@REM -renameSelectedComponent component123 ^
@REM -exportXMPForSelectedComponent ^
@REM -exportSelectedComponent "%workingDir%\components\\"

rem Waits for the last preceding process to be finished before it continues
call waitCompleted.bat 

@REM rem Cycle which moves XMP files from folders images1/2/3 to the folder cameras123
@REM FOR /L %%I IN (1,1,3) DO (
@REM     MOVE /Y "%workingDir%\images%%I\*.xmp" "%workingDir%\cameras123\\"
@REM )
