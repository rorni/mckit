call conda deactivate
call conda env remove -n mckit4 -q -y
call conda create -n mckit4 python=3.8 -q -y
call conda activate mckit4
pip install mkl-devel numpy scipy scikit-learn numexpr
echo The following two commands should point to the same environment
call conda env list
call poetry env info --path
python build_nlopt.py
call poetry build
call poetry install
pytest -m "not slow"
:: tornado (in jupyter) doesn't work with newer version of pywin, check this on jupyter dependencies updates
call conda install pywin32=227 -y
call poetry run python -m ipykernel install --user --name mckit4
mckit --version
nox --list
nox -s safety
nox -s tests  -- -m "not slow" --cov

