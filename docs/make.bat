@ECHO OFF

pushd %~dp0
if "%BUN%" == "" set BUN=bun
%BUN% run --cwd ..\src\typescript docs:build
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
