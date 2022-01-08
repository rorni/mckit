@echo off
::
:: Prepare conda environment for mckit development on Windows.
::
:: dvp, Dec 2020
::

set mckit_version=5.11

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
if "%mckit%"=="" set mckit=mckit-%mckit_version%


set install_tool=%1
shift
if "%install_tool%"=="" set install_tool=pip

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

git submodule update --recursive --depth=1
:: 1) Conda downgrades mkl, so we use pip instead (ant it works fine for us)
:: 2) We need numpy to build nlopt, other packages are installed just for convenience
:: pip install mkl-devel numpy scipy scikit-learn numexpr


:: install mckit to the current environment
if "%install_tool%"=="pip" (
    pip install .
    pip install -r requirements-dev.txt
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
    nox -s safety -s tests -p %python_version% -- -m "not slow" --cov
    call poetry build
    if errorlevel 1 (
        echo ERROR: failed to run poetry build
        goto END
    )
) else (
    pip install .
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
