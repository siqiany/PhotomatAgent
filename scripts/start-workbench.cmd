@echo off
setlocal

where wsl.exe >nul 2>nul
if errorlevel 1 (
  echo 找不到 WSL。请先安装并启动一个 WSL Linux 发行版。 1>&2
  exit /b 2
)

rem Pass the script directory as an argv value; wslpath handles Windows paths,
rem including spaces and Unicode, inside the selected WSL distribution.
wsl.exe -e bash -lc "root=$1; shift; cd -- \"$(wslpath -u \"$root\")\" && exec bash scripts/start-workbench.sh \"$@\"" start-workbench "%~dp0.." %*
exit /b %errorlevel%
