call conda deactivate
call conda env remove -n mckit4 -q -y
call conda create -n mckit4 python=3.8 -q -y
call conda activate mckit4
pip install mkl-devel numpy scipy scikit-learn numexpr
echo The following two commands should point to the same environment
call conda list
call poetry env info --path
python build_nlopt.py
call poetry build
call poetry install
pytest -m "not slow"
call poetry run ipython kernel install --user --name=mckit4
mckit --version
nox --list
:: nox -s safety
:: nox -s tests

