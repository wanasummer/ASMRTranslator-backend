@echo off
rem Build the extracted FSMN-VAD on Windows (MSVC or MinGW + CMake required).
rem Produces build\bin\fsmn_vad.dll (C API) and build\bin\llama-funasr-vad.exe (CLI).
setlocal
cd /d "%~dp0"
if not exist "build" mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build -j --config Release
if errorlevel 1 exit /b 1
echo.
echo Done. Outputs:
dir /b build\bin\*.dll build\bin\*.exe 2>nul
endlocal
