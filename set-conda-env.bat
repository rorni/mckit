::
:: Prepare conda environment for mckit development on Windows.
::
:: dvp, Dec 2020
::

set mckit=%1

if "%mckit%"=="" (
    set mckit=mckit
)

echo Installing conda/mckit environment %mckit%

call conda deactivate
call conda activate
call conda env remove -n %mckit% -q -y
call conda create -n %mckit% python=3.8 -q -y
call conda activate %mckit%

:: 1) Conda downgrades mkl, so we use pip instead (ant it works fine for us)
:: 2) We need numpy to build nlopt, other packages are installed just for convenience
pip install mkl-devel numpy scipy scikit-learn numexpr

:: The following two commands should point to the same environment
call conda env list
call poetry env info --path
echo .
echo Check the previous two outputs: do they point to the same environment?
echo ---------------------------------------------------------------------
echo Set IDE (VSCode, PyCharm etc.) to use this environment.
echo .

:: build and install nlopt to the current environment (for developers only, requires CMake)
:: This is called automatically on poetry build
:: python build_nlopt.py

:: install mckit to the current environment
call poetry install

:: build mckit itself
:: This is called 
:: call poetry build


:: check if mckit is actually installed and everything is OK with dependencies
mckit --version

:: verify pytest
pytest -m "not slow"

:: Jupyter setup
::
:: tornado (in jupyter) doesn't work with newer version of pywin, check this on jupyter dependencies updates
:: TODO dvp: check on dependencies updates
::
call conda install pywin32=227 -y
call poetry run python -m ipykernel install --user --name %mckit%
:: To use this environment in jupyter:
:: - Run 'jupyter lab'  (note: 'jupyter notebook' is deprecated, but works so far)
:: - Open or create notebook
:: - Select kernel %mckit%
:: - check if 'import mckit' in the notebook works

:: verify nox
nox --list
:: safety first - run this on every dependency addition or update
nox -s safety
:: test often - not tested commits is a road to hell
:: TODO dvp: nox fails on Windows after changes in 0.5.1
::      however it worked fine in 0.4
::      Fix this: probably use environment setting as
::      it was done for LD_LIBRARY_PATH
:: nox -s tests -p 3.8 -- -m "not slow" --cov

:: verify if 'pip' is able to collect the dependencies wheels
:: Note:
::     'poetry build' (see above) creates the wheel for mckit
::     So, this command is not necessary, just demo how does 'pip' work without setup.py.
:: pip wheel -w dist --verbose .
