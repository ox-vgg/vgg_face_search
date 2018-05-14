@echo off
REM Parameters:
REM %1 -> Type of input: "video" or "images"
REM %2 -> Full path to video (if input type is "video") or full path to text file containing list of images to ingest (if input type is "images")
REM %3 -> Full path to base folder holding the images referenced by the search service
REM %4 -> Full path to output features file (optional)
cd "%~dp0"
REM activate virtual env
call ..\..\Scripts\activate
setlocal enableextensions enabledelayedexpansion
if "%1"=="video" (
  REM get video name
  set VIDEONAME=%~n2
  if NOT  "!VIDEONAME!"=="" (
      REM remove previous temporary folder/file, if present
      rmdir /S /Q "%~dp0..\..\tmp\!VIDEONAME!"
      del /Q "%~dp0..\..\tmp\!VIDEONAME!_*.txt"
      REM make new temporary folder
      mkdir "%~dp0..\..\tmp\!VIDEONAME!"
      REM extract video fps value
      ffmpeg -i %2 > "%~dp0..\..\tmp\!VIDEONAME!_ffmpeg.txt" 2>&1
      powershell.exe -nologo -noprofile -command "cat '%~dp0..\..\tmp\!VIDEONAME!_ffmpeg.txt' | %%{[Regex]::Matches($_,'.. fps')} | %%{$_.Value} | %%{$_ -replace ' fps', '' } | Out-File -encoding ascii '%~dp0..\..\tmp\!VIDEONAME!_fps.txt'"
      SET /p FPS=<"%~dp0..\..\tmp\!VIDEONAME!_fps.txt"
      REM extract all video frames and run shot detection
      ffmpeg -i %2 -vsync vfr -q:v 1 -start_number 0 -vf "scale=iw:ih*(1/sar)" -loglevel panic "%~dp0..\..\tmp\!VIDEONAME!\%%05d.jpg"
      python detect_shots\detect_shots.py "%~dp0..\..\tmp\!VIDEONAME!" "%~dp0..\..\tmp\!VIDEONAME!_shots.txt" -f !FPS! -s
      REM remove frames and re-extract them, but only 1 frame per second
      rmdir /S /Q "%~dp0..\..\tmp\!VIDEONAME!"
      mkdir "%~dp0..\..\tmp\!VIDEONAME!"
      REM from 0 to 90 minutes
      for /L %%i in (0, 1, 5400) do (
       set "formattedValue=00000%%i"
       set FNAME=!formattedValue:~-5!
       ffmpeg -ss %%i -i %2 -vframes 1 -q:v 1 -vf "scale=iw:ih*(1/sar)" -loglevel panic "%~dp0..\..\tmp\!VIDEONAME!\!FNAME!.jpg"
       if NOT exist "%~dp0..\..\tmp\!VIDEONAME!\!FNAME!.jpg" goto :next
      )
      REM endlocal
      :next
      if "%~4"=="" (
           python compute_pos_features_video.py "%~dp0..\..\tmp\!VIDEONAME!" "%~dp0..\..\tmp\!VIDEONAME!_shots.txt" "%3"
      ) else (
           python compute_pos_features_video.py "%~dp0..\..\tmp\!VIDEONAME!" "%~dp0..\..\tmp\!VIDEONAME!_shots.txt" "%3" -o "%4"
      )
      REM clean up
      rmdir /S /Q "%~dp0..\..\tmp\!VIDEONAME!"
      del /Q "%~dp0..\..\tmp\!VIDEONAME!_*.txt"
  )
  goto :end
) else (
    if "%~4"=="" (
        python compute_pos_features.py "%2" "%3"
    ) else (
        python compute_pos_features.py "%2" "%3" -o "%4"
    )
)
:end
endlocal
