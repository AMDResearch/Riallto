@echo off

REM Step 1: Build the Jupyter Book
jupyter-book build .

REM Step 2: Copy files from ../images to ./dir/images
xcopy /s /i /y ".\notebooks\images\*" ".\_build\html\notebooks\images"

echo. > ".\_build\html\.nojekyll"