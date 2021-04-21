

Install
-------

From source: ::

    git clone git@github.com:rorni/mckit.git
    cd mckit

    # set local python environment with `pyenv`, `venv` or `conda`
    # pyenv virtualenv 3.9.1 mckit
    # pyenv local mckit 3.9.1 3.8.5 3.7.9 3.6.12
    # or (conda is better for Windows)
    # conda create -n mckit python=3.9
    # conda activate mckit

    # if you have `poetry` installed
    poetry build
    poetry install

    # either
    pip install .

There are also scripts to setup virtual environment: ::

    - Linux:   reset-pyenv-env.sh
    - Windows: reset-conda-env.bat

From wheel: ::

    pip install <wheel>


From PyPI: ::

    pip install mckit

Tutorial
--------
See the notebook in ``tutorial`` folder.

Documentation
--------------
.. TODO add reference to readthedocs

TODO
~~~~
.. TODO add nearest future plans.

* translate documentation to English
* implement generation of the documentation for `readthedocs`
* create and link to the `readthedocs` account
* add the `readthedocs` status icon at the document start.

A Developer's Reading
----------------------
* `The Hitchhicker's guide to Python <https://docs.python-guide.org/>`_
* `Claudio Cjolowicz "Hypermodern Python" <https://cjolowicz.github.io/posts/hypermodern-python-01-setup>`_
* `Python development best practices 2019 <https://flynn.gg/blog/software-best-practices-python-2019/>`_
* `Managing Multiple Python Versions With pyenv <https://realpython.com/intro-to-pyenv/>`_
* `Poetry  <https://python-poetry.org/>`_
* `Pytest <https://docs.pytest.org/en/stable/index.html>`_
* `Linting <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
* `Sphinx, Стандартный синтаксис разметки  reStructuredText, readthedocs <https://sphinx-ru.readthedocs.io/ru/latest/rst-markup.html>`_
* `Python Packaging Guide <https://packaging.python.org>`_
* `Stop using Anaconda <https://medium.com/swlh/stop-using-anaconda-for-your-data-science-projects-1fc29821c6f6>`_

Resources
---------
* `Budges for README.rst <https://github.com/Naereen/badges/blob/master/README.rst>`_
* `About commit message format <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit>`_
* `Semantic Versioning <https://semver.org/>`_
* `Typing <https://www.python.org/dev/peps/pep-0484/>`_
* `Why pyproject.toml <https://www.python.org/dev/peps/pep-0518/>`_
* `Git branching and tagging best practices <https://nvie.com/posts/a-successful-git-branching-model/>`_

Check if we can apply these packaging tools
-------------------------------------------
* `Packaging Tutorial: <https://python-packaging-tutorial.readthedocs.io/en/latest/binaries_dependencies.html>`_
* `scikit-build <https://scikit-build.readthedocs.io/en/latest/index.html>`_
* `Benjamin R. Jack, Hybrid Python/C++ packages, revisited <https://www.benjack.io/2018/02/02/python-cpp-revisited.html>`_

Bugs
----

Please report new bugs via the `Github issue tracker <https://github.com/rorni/mckit/issues>`_.


Development
-----------

For Linux we assume usage of pyenv/poetry toolchain.

.. TODO explain details for activation of development environment
.. TODO add MKL handling stuff.

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

To provide proper change logs, apply this format for commit messages::

    <type>: <short summary>
      │       │
      │       └─⫸ Summary in present tense. Not capitalized. No period at the end.
      │
      └─⫸ Commit Type: breaking|build|ci|doc|feature|bug|performance|refactoring|removal|style|test


.. list-table:: Commit types description
    :widths: 20 30
    :header-rows: 1

    * - Commit Type
      - Description
    * - breaking
      - Breaking changes introducing API incompatibility
    * - build
      - Build System
    * - ci
      - Continuous Integration'
    * - doc
      - Documentation
    * - feature
      - Features change to satisfy tests
    * - bug
      - Fixes bug, no other changes in the code
    * - performance
      - Performance, benchmarks or profiling changes.
    * - refactoring
      - Refactoring code without changes in features and tests
    * - removal
      - Removing and deprecations in code or dependencies
    * - style
      - Code and documentation style improvements. No changes in tests and features.
    * - test
      - Changes in tests without adding features
