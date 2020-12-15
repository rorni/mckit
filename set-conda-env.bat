conda update conda
conda update anaconda
conda config --add channels intel
conda create -n mckit4 python=3.8
conda activate mckit4
conda install mkl-devel
conda list
poetry env info --path
python build_nlopt.py
poetry build
pytest
poetry install
mckit --version
nox --list
nox -s safety
nox -s tests

