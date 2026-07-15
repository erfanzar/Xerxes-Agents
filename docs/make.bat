@ECHO OFF

pushd %~dp0
if "%BUN%" == "" set BUN=bun
%BUN% run --cwd ..\xerxes docs:build
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
