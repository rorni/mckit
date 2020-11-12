# 
# See https://github.com/pyenv-win/pyenv-win and https://python-poetry.org/docs/
#
[System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
[System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
[System.Environment]::SetEnvironmentVariable('path', $HOME + "\.pyenv\pyenv-win\bin;" + $HOME + "\.pyenv\pyenv-win\shims;" + $HOME + "\.poetry\bin;"+ $env:Path,"User")
git clone https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"
pyenv install 3.7.7
pyenv install 3.8.2
pyenv global 3.8.2 3.8.7  # These are the most recent versions available for 2020-11-12, update as needed
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
