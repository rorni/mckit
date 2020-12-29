@echo off
::
:: Prepare conda environment for mckit development on Windows.
::
:: dvp, Dec 2020
::

set mckit_version=5.1

if "%1"=="--help" (
    echo.
    echo Usage:
    echo.
    echo set-conda-env conda_env install_tool python_version
    echo.
    echo All the parameters are optional.
    echo.
    echo Defaults:
    echo   conda_env=mckit
    echo   install_tool=pip   another valid value: poetry
    echo   python_version=3.9
    echo.
    goto END
)

set mckit=%1
shift
if "%mckit%"=="" set mckit=mckit%mckit_version%


set install_tool=%1
shift
if "%install_tool%"=="" set install_tool=poetry

if "%install_tool%"=="poetry" (
    call poetry --version > NUL
    if errorlevel 1 (
        echo ERROR\: Poetry is not available
        echo        See poetry install instructions: https://python-poetry.org
        goto END
    )
) else (
    if "%install_tool%" NEQ "pip" (
        echo ERROR\: unknown install tool %install_tool%. Should be either `pip` or `poetry`
        goto END
    )
)


set python_version=%1
shift
if "%python_version%"=="" (
    set python_version=3.9
)


echo Installing conda environment %mckit% with %install_tool%

call conda deactivate
call conda activate
call conda env remove -n %mckit% -q -y
call conda create -n %mckit% python=%python_version% -q -y
call conda activate %mckit%

if "%install_tool%"=="pip" (
    ::   this actually installs the package to the environment
    :: Note
    ::   pip install . installs mckit with wrong version of pyd module: for the oldest python
    ::   in the range specified in pyproject.toml
    ::   So, you need poetry anyway:
    del dist\mckit*amd64.whl
    call poetry build
    call poetry export --without-hashes --format requirements.txt --dev > requirements-dev.txt
    for %%f in ( dist\mckit*amd64.whl )  do (
        pip install %%f
        pip install -r requirements-dev.txt
    )
) else (
    ::   this creates egg-link in the environment to current directory (development install)
    call poetry install
)
if errorlevel 1  (
    echo "ERROR: failed to run install with %install_tool%"
    goto END
)

mckit --version
if errorlevel 1  (
    echo "ERROR: failed to install mckit"
    goto END
)
echo.
echo SUCCESS: mckit has been installed
echo.


pytest -m "not slow"
if errorlevel 1 (
    echo ERROR: failed to run tests
    goto END
)
echo.
echo SUCCESS: pytest is passed OK
echo.



if "%install_tool%"=="poetry" (
    :: verify nox
    nox --list
    :: safety first - run this on every dependency addition or update
    :: test often - who doesn't?
    nox -s safety -s tests -p 3.9 -- -m "not slow" --cov
    call poetry build
    if errorlevel 1 (
        echo ERROR: failed to run poetry build
        goto END
    )
) else (
    :: verify if 'pip' is able to collect the dependencies wheels
    pip wheel -w dist .
    if errorlevel 1 (
        echo ERROR: failed to collect dependencies with pip
        goto END
    )
)

call create-jk %mckit%
if errorlevel 1 (
    goto END
)

echo.
echo SUCCESS!
echo --------
echo.
echo Usage:
echo.
mckit --help
echo.
echo Conda environment %mckit% is all clear.
echo Set your IDE to use %CONDA_PREFIX%\python.exe
echo.
echo Enjoy!
echo.


:END
