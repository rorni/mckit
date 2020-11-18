.. include:: preamble.rst

.. |Maintenance yes| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/rorni/mckit/graphs/commit-activity

.. |Tests| image:: https://github.com/rorni/mckit/workflows/Tests/badge.svg
   :target: https://github.com/rorni/mckit/actions?workflow=Tests

.. |Codecov| image:: https://codecov.io/gh/rorni/mckit/branch/master/graph/badge.svg)
    :target: https://codecov.io/gh/rorni/mckit

.. |PyPI| image:: https://img.shields.io/pypi/v/mckit.svg
   :target: https://pypi.org/project/mckit/

.. |Read the Docs| image:: https://readthedocs.org/projects/mckit/badge/
   :target: https://mckit.readthedocs.io/

MCKIT
=====

Introduction
------------

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

With setup.py: ::

    git clone git@github.com:rorni/mckit.git
    cd mckit
    python setup.py build
    python setup.py install

Copyright
---------
|mckit| is free software.
TODO add dependencies licenses here


Contributors
------------
* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_
* `Dmitry Portnov <mailto:dmitri_portnov@yahoo.com>`_

Copyright
---------
|mckit| is free software.
.. TODO add dependencies licenses here


Tutorial
--------
See the notebook in ``tutorial`` folder.

Documentation
--------------
.. TODO add reference to readthedocs

TODO
~~~~
.. TODO add nearest future plans.

* implement generation of the documentation for readthedocs
* create and link to the readthedocs account
* add the readthedocs status icon at the document start.

A Developmer's Reading
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

Resources
---------
* `Budges for README.rst <https://github.com/Naereen/badges/blob/master/README.rst>`_
* `Commit message format <https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit>`_
* `Semantic Versioning <https://semver.org/>`_
* `Typing <https://www.python.org/dev/peps/pep-0484/>`_

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
      └─⫸ Commit Type: breaking|build|ci|documentation|enhancement|bug|performance|refactoring|removal|style|testing


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
    * - enhancement
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

