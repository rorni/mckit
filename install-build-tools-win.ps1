#
# poetry can be installed with this script
#
# 
# See https://github.com/pyenv-win/pyenv-win and https://python-poetry.org/docs/
#

# If we decide to use pyenv-win
# [System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
# [System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
# [System.Environment]::SetEnvironmentVariable('path', $HOME + "\.pyenv\pyenv-win\bin;" + $HOME + "\.pyenv\pyenv-win\shims;" + $HOME + "\.poetry\bin;"+ $env:Path,"User")
# git clone https://github.com/pyenv-win/pyenv-win.git "$HOME/.pyenv"
# pyenv update
# pyenv install 3.7.9
# pyenv install 3.8.6
# pyenv install 3.9.0
# 
# Note: pyenv-win doesn't provide 'system' option for global command
#
# pyenv global 3.9.0 3.8.6 3.7.9  # These are the most recent versions available for 2020-11-12, update as needed
# pyenv virtualenv 3.8.2 mckit0.5.0-py3.8
# pyenv local mckit0.5.0-py3.8 3.9 3.8 3.7


# Poetry
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -

