.. include:: preamble.rst

.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/dvp2015/mckit/graphs/commit-activity

.. image:: https://github.com/dvp2015/mckit/workflows/Tests/badge.svg
   :target: https://github.com/dvp2015/mckit/actions?workflow=Tests

.. |Codecov| image:: https://codecov.io/gh/rorni/mckit/branch/master/graph/badge.svg)
    :target: https://codecov.io/gh/rorni/mckit

.. |PyPI| image:: https://img.shields.io/pypi/v/mckit.svg
   :target: https://pypi.org/project/mckit/

.. |Read the Docs| image:: https://readthedocs.org/projects/mckit/badge/
   :target: https://mckit.readthedocs.io/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/

.. image:: https://img.shields.io/github/license/dvp2015/mckit
   :target: https://github.com/dvp2015/mckit



MCKIT: MCNP model and results processing utilites
=================================================

Purpose
-------

The package |mckit| provides library to work with
MCNP models and results. The package also provides command line interface, which 
allows compose and decompose a MCNP model over universes hierarchy, split a model
to text portions and combine back.

.. tip::

   To see actual list of available commands run::

       mckit --help

.. TODO dvp implement pre-commit hook to print the CLI help into file, format it for as reStructuredText and include here.

The library can be used for more complicated and specific tasks.

Install
-------

From source: ::

    git clone git@github.com:rorni/mckit.git
    cd mckit

    # set local python environment with `pyenv`, `venv` or `conda`
    # pyenv virtualenv 3.9.1 mckit
    # pyenv local mckit 3.9.1 3.8.5 3.7.9 3.6.12
    # or
    # conda create -n mckit python=3.9
    # conda activate mckit

    # if you have `poetry` installed
    poetry build
    poetry install

    # either (without poetry)
    pip install .


From wheel: ::

    pip install <wheel>


From PyPI: ::

    To be implemented


Copyright
---------
|mckit| is free software.

The dependencies are declared in the package meta-information.


Contributors
------------
* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_
* `Dmitry Portnov <mailto:dmitri_portnov@yahoo.com>`_


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
* `Commit message format <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit>`_
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
      └─⫸ Commit Type: breaking|build|ci|documentation|feature|bug|performance|refactoring|removal|style|testing


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
    * - documentation
      - Documentation
    * - feature
      - Features
    * - bug
      - Fixes bug, no other changes in the code
    * - performance
      - Performance, benchmarks or profiling changes.
    * - refactoring
      - Refactoring
    * - removal
      - Removing and deprecations in code or dependencies
    * - style
      - Code and documentation style improvements.
    * - testing
      - Changes in tests

