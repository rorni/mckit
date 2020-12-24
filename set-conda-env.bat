@echo off
::
:: Prepare conda environment for mckit development on Windows.
::
:: dvp, Dec 2020
::

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
if "%mckit%"=="" set mckit=mckit


set install_tool=%1
shift
if "%install_tool%"=="" set install_tool=pip

if "%install_tool%"=="pip" (
    echo Installing %mckit% with pip
) else (
    if "%install_tool%"=="poetry" (
        call poetry --version > NUL
        if not errorlevel 0 (
            echo ERROR\: Poetry is not available
            echo        See poetry install instructions: https://python-poetry.org
            goto END
        )
    ) else (
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

:: 1) Conda downgrades mkl, so we use pip instead (ant it works fine for us)
:: 2) We need numpy to build nlopt, other packages are installed just for convenience
:: pip install mkl-devel numpy scipy scikit-learn numexpr


:: install mckit to the current environment
if "%install_tool%"=="pip" (
    pip install .
) else (
    :: In development environment use
    :: pip install -e
    :: or, if poetry is available (this is preferable)
    :: The following two commands should point to the same environment
    call conda env list
    call poetry env info --path
    echo.
    echo Check the previous two outputs\: do they point to the same environment?
    echo ----------------------------------------------------------------------
    echo.
    call poetry install
)
mckit --version
if errorlevel 0  goto SUCCESS1

echo "ERROR: failed to install mckit"
goto END


:SUCCESS1
echo.
echo SUCCESS: mckit has been installed
echo.
pytest -m "not slow"
if errorlevel 0 goto SUCCESS2

echo ERROR: failed to run tests
goto END

:SUCCESS2
echo.
echo SUCCESS: pytest is passed OK
echo.


if "%install_tool%"=="poetry" (
    :: verify nox
    nox --list
    :: safety first - run this on every dependency addition or update
    :: test often - who doesn't?
    nox -s safety,tests -p 3.9 -- -m "not slow" --cov
    call poetry build
) else (
    :: verify if 'pip' is able to collect the dependencies wheels
    pip wheel -w dist .
)

if errorlevel 0 goto SUCCESS3

echo ERROR: failed to collect dependencies with pip
goto END

:SUCCESS3
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
call create-jk %mckit%

:END
