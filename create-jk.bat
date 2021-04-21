@echo off
::
:: Jupyter kernel setup
::
::
:: dvp, Dec 2020
::

set mckit=%1

if "%mckit%"=="" (
    set mckit=mckit
)

echo Creating jupyter kernel for conda environment %mckit%

:: Fix pywin32 version for tornado
:: tornado (in jupyter) doesn't work with newer version of pywin, check this on jupyter dependencies updates
:: TODO dvp: check on dependencies updates
:: The following sets version 228 on python39 (after pip or poetry it was 300)
:: call conda install pywin32 -y
call conda install jupyterlab -y

:: Create jupyter kernel pointing to the conda environment
call python -m ipykernel install --user --name %mckit%
if errorlevel 1 (
    echo ERROR: something wrong with installing Jupyter kernel for %mckit% environment
    set errorlevel=1
) else (
    echo To use %mckit% environment in jupyter
    echo   - Run 'jupyter lab'
    echo   - Open or create notebook
    echo   - Select kernel %mckit%
    echo   - check if 'import mckit' in the notebook works
)
