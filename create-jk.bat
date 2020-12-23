%% set jupyter kernel for conda environment
:: Jupyter setup
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
call pip install pywin32=227 -y  

:: Create jupyter kernel pointing to the conda environment
call poetry run python -m ipykernel install --user --name %mckit%

:: To use this environment in jupyter:
:: - Run 'jupyter lab'  (note: 'jupyter notebook' is deprecated, but works so far)
:: - Open or create notebook
:: - Select kernel %mckit%
:: - check if 'import mckit' in the notebook works
