# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['mckit',
 'mckit.cli',
 'mckit.cli.commands',
 'mckit.parser',
 'mckit.parser.common',
 'mckit.utils']

package_data = \
{'': ['*'], 'mckit': ['data/*', 'src/*']}

install_requires = \
['DateTime>=4.3,<5.0',
 'Pygments>=2.7.2,<3.0.0',
 'aiofile>=3.3.3,<4.0.0',
 'asyncio>=3.4.3,<4.0.0',
 'atomicwrites>=1.4.0,<2.0.0',
 'attrs>=20.2.0,<21.0.0',
 'click-loguru>=1.3.4,<2.0.0',
 'click>=7.1.2,<8.0.0',
 'colorama>=0.4.4,<0.5.0',
 'isort>=5.6.4,<6.0.0',
 'mkl-devel>=2021.1.1,<2022.0.0',
 'mkl-include>=2021.1.1,<2022.0.0',
 'mkl>=2021.1.1,<2022.0.0',
 'numpy>=1.19.4,<2.0.0',
 'numpydoc>=1.1.0,<2.0.0',
 'ply>=3.11,<4.0',
 'python-dotenv>=0.15.0,<0.16.0',
 'scipy>=1.5.3,<2.0.0',
 'sly>=0.4,<0.5',
 'tomlkit>=0.7.0,<0.8.0',
 'toolz>=0.11.1,<0.12.0',
 'tqdm>=4.53.0,<5.0.0']

extras_require = \
{':python_version < "3.8"': ['importlib-metadata>=2.0.0,<3.0.0'],
 ':python_version < "3.8" and sys_platform == "Windows"': ['certifi==2020.6.20']}

entry_points = \
{'console_scripts': ['mckit = mckit.cli.runner:mckit']}

setup_kwargs = {
    'name': 'mckit',
    'version': '0.5.1a0',
    'description': 'Tools to process MCNP models and results',
    'long_description': '.. |copy| unicode:: 0xA9 .. copyright\n.. |(TM)| unicode:: U+2122 .. trademark\n.. |---| unicode:: U+02014 .. long dash\n.. |date| date:: %d.%m.%Y\n.. |time| date:: %H:%M\n.. |mckit| replace:: ``mckit``\n.. |br| raw:: html\n\n       <br />\n\n.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg\n   :target: https://github.com/dvp2015/mckit/graphs/commit-activity\n\n.. image:: https://github.com/dvp2015/mckit/workflows/Tests/badge.svg\n   :target: https://github.com/dvp2015/mckit/actions?workflow=Tests\n\n\n.. image:: https://codecov.io/gh/dvp2015/mckit/branch/devel/graph/badge.svg?token=05OFBQS3RX\n   :target: https://codecov.io/gh/dvp2015/mckit\n\n\n.. |PyPI| image:: https://img.shields.io/pypi/v/mckit.svg\n   :target: https://pypi.org/project/mckit/\n\n.. |Read the Docs| image:: https://readthedocs.org/projects/mckit/badge/\n   :target: https://mckit.readthedocs.io/\n\n.. image:: https://img.shields.io/badge/code%20style-black-000000.svg\n    :target: https://github.com/psf/black\n\n.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336\n    :target: https://pycqa.github.io/isort/\n\n.. image:: https://img.shields.io/github/license/dvp2015/mckit\n   :target: https://github.com/dvp2015/mckit\n\n\n\nMCKIT: MCNP model and results processing utilites\n=================================================\n\nPurpose\n-------\n\nThe package |mckit| provides library to work with\nMCNP models and results. The package also provides command line interface, which \nallows compose and decompose a MCNP model over universes hierarchy, split a model\nto text portions and combine back.\n\n.. tip::\n\n   To see actual list of available commands run::\n\n       mckit --help\n\n.. TODO dvp implement pre-commit hook to print the CLI help into file, format it for as reStructuredText and include here.\n\nThe library can be used for more complicated and specific tasks.\n\nInstall\n-------\n\nFrom source: ::\n\n    git clone git@github.com:rorni/mckit.git\n    cd mckit\n\n    # set local python environment with `pyenv`, `venv` or `conda`\n    # pyenv virtualenv 3.9.1 mckit\n    # pyenv local mckit 3.9.1 3.8.5 3.7.9 3.6.12\n    # or (conda is better for Windows)\n    # conda create -n mckit python=3.9\n    # conda activate mckit\n\n    # if you have `poetry` installed\n    poetry build\n    poetry install\n\n    # either\n    pip install .\n\nThere are also scripts to setup virtual environment: ::\n    - Linux:   reset-pyenv-env.sh\n    - Windows: reset-conda-env.bat\n\nFrom wheel: ::\n\n    pip install <wheel>\n\n\nFrom PyPI: ::\n\n    pip install mckit\n\n\nCopyright\n---------\n|mckit| is free software.\n\nThe dependencies are declared in the package meta-information.\n\n\nContributors\n------------\n* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_\n* `Dmitry Portnov <mailto:dmitri_portnov@yahoo.com>`_\n\n\nTutorial\n--------\nSee the notebook in ``tutorial`` folder.\n\nDocumentation\n--------------\n.. TODO add reference to readthedocs\n\nTODO\n~~~~\n.. TODO add nearest future plans.\n\n* translate documentation to English\n* implement generation of the documentation for `readthedocs`\n* create and link to the `readthedocs` account\n* add the `readthedocs` status icon at the document start.\n\nA Developer\'s Reading\n----------------------\n* `The Hitchhicker\'s guide to Python <https://docs.python-guide.org/>`_ \n* `Claudio Cjolowicz "Hypermodern Python" <https://cjolowicz.github.io/posts/hypermodern-python-01-setup>`_\n* `Python development best practices 2019 <https://flynn.gg/blog/software-best-practices-python-2019/>`_\n* `Managing Multiple Python Versions With pyenv <https://realpython.com/intro-to-pyenv/>`_\n* `Poetry  <https://python-poetry.org/>`_\n* `Pytest <https://docs.pytest.org/en/stable/index.html>`_\n* `Linting <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_\n* `Sphinx, Стандартный синтаксис разметки  reStructuredText, readthedocs <https://sphinx-ru.readthedocs.io/ru/latest/rst-markup.html>`_\n* `Python Packaging Guide <https://packaging.python.org>`_\n* `Stop using Anaconda <https://medium.com/swlh/stop-using-anaconda-for-your-data-science-projects-1fc29821c6f6>`_\n\nResources\n---------\n* `Budges for README.rst <https://github.com/Naereen/badges/blob/master/README.rst>`_\n* `Commit message format <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit>`_\n* `Semantic Versioning <https://semver.org/>`_\n* `Typing <https://www.python.org/dev/peps/pep-0484/>`_\n* `Why pyproject.toml <https://www.python.org/dev/peps/pep-0518/>`_\n* `Git branching and tagging best practices <https://nvie.com/posts/a-successful-git-branching-model/>`_\n\nCheck if we can apply these packaging tools\n-------------------------------------------\n* `Packaging Tutorial: <https://python-packaging-tutorial.readthedocs.io/en/latest/binaries_dependencies.html>`_\n* `scikit-build <https://scikit-build.readthedocs.io/en/latest/index.html>`_\n* `Benjamin R. Jack, Hybrid Python/C++ packages, revisited <https://www.benjack.io/2018/02/02/python-cpp-revisited.html>`_\n\nBugs\n----\n\nPlease report new bugs via the `Github issue tracker <https://github.com/rorni/mckit/issues>`_.\n\n\nDevelopment\n-----------\n\nFor Linux we assume usage of pyenv/poetry toolchain.\n\n.. TODO explain details for activation of development environment\n.. TODO add MKL handling stuff.\n\nCommit Message Format\n~~~~~~~~~~~~~~~~~~~~~\n\nTo provide proper change logs, apply this format for commit messages::\n\n    <type>: <short summary>\n      │       │\n      │       └─⫸ Summary in present tense. Not capitalized. No period at the end.\n      │\n      └─⫸ Commit Type: breaking|build|ci|doc|feature|bug|performance|refactoring|removal|style|test\n\n\n.. list-table:: Commit types description\n    :widths: 20 30\n    :header-rows: 1\n\n    * - Commit Type\n      - Description\n    * - breaking\n      - Breaking changes introducing API incompatibility\n    * - build\n      - Build System\n    * - ci\n      - Continuous Integration\'\n    * - doc\n      - Documentation\n    * - feature\n      - Features change to satisfy tests\n    * - bug\n      - Fixes bug, no other changes in the code\n    * - performance\n      - Performance, benchmarks or profiling changes.\n    * - refactoring\n      - Refactoring code without changes in features and tests\n    * - removal\n      - Removing and deprecations in code or dependencies\n    * - style\n      - Code and documentation style improvements. No changes in tests and features.\n    * - test\n      - Changes in tests without adding features\n\n',
    'author': 'rrn',
    'author_email': 'r.rodionov@iterrf.ru',
    'maintainer': 'dpv2015',
    'maintainer_email': 'dmitri_portnov@yahoo.com',
    'url': 'https://github.com/rorni/mckit',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.7,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
