@echo off
chcp 65001
set "cpath=%~dp0"
set "cpath=%cpath:~0,-1%"
set "suffixes=mp3,wav,flac,ogg"
if exist "%cpath%\infer.exe" (
  "%cpath%\infer.exe" --audio_suffixes="%suffixes%" --sub_formats="srt,vtt,lrc" --device="cuda" --output_dir="输出" %*
) else (
  python "%cpath%\infer.py" --audio_suffixes="%suffixes%" --sub_formats="srt,vtt,lrc" --device="cuda" --output_dir="输出" %*
)
pause
