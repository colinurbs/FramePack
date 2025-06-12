@echo off
echo FramePack-Studio Update Script

REM Check if Git is installed (basic check)
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in your PATH. Unable to update.
    goto end
)

REM Check if Python is installed (basic check)
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH. Unable to update dependencies.
    REM Continue with Git pull, but warn about dependencies
    echo Warning: Python is not available, skipping dependency update.
    goto git_pull
)


:git_pull
echo Pulling latest changes from Git...
git pull

REM Check if git pull was successful
if %errorlevel% neq 0 (
    echo Error: Failed to pull latest changes from Git. Please resolve any conflicts manually.
    goto end
)

echo Git pull successful.

REM Attempt to update dependencies if Virtual Environment is available
if exist "%cd%/venv/Scripts/python.exe" (

if %errorlevel% equ 0 (
    echo Updating dependencies using pip...
    REM This assumes there's a requirements.txt file in the root
    REM Using --upgrade to update existing packages
    "%cd%/venv/Scripts/python.exe" -m pip install --upgrade -r requirements.txt

    REM Check if pip update was successful
    if %errorlevel% neq 0 (
        echo Warning: Failed to update dependencies. You may need to update them manually.
    ) else (
        echo Dependency update successful.
    )
) else (
    echo Skipping dependency update as Python is not available.
)

) else (

echo Error: Virtual Environment for Python not found. Did you install correctly?
goto end 

)



echo Update complete.

:end
echo Exiting update script.
pause