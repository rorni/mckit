.. include:: preamble.rst

.. |Maintenance yes| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/rorni/mckit/graphs/commit-activity

.. |Tests| image:: https://github.com/rorni/mckit/workflows/Tests/badge.svg
   :target https://github.com/rorni/mckit/actions?workflow=Tests

MCKIT
=====

Introduction
------------

The package defines Python package |mckit|, which provides library to work with
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
TODO add dependecies licenses here


Contributors
------------
* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_
* `Dmitry Portnov <mailto:dmitri_portnov@yahoo.com>`_

Copyright
---------
|mckit| is free software.
TODO add dependecies licenses here


Authors
-------
* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_

Maintainers
-----------
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

* implement generation of the documentation for readthedocs
* create and link to the readocs account
* add the readthedocs status icon at the document start.

Good reading
------------
* `Build and dependencies management <https://cjolowicz.github.io/posts/hypermodern-python-01-setup>`_
* `Python development best practices 2019 <https://flynn.gg/blog/software-best-practices-python-2019/>`_
* `Managing Multiple Python Versions With pyenv <https://realpython.com/intro-to-pyenv/>`_
* `Poetry  <https://python-poetry.org/>`_
* `Pytest <https://docs.pytest.org/en/stable/index.html>`_
* `Linting <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
* `Sphinx, Стандартный синтаксис разметки  reStructuredText, readthedocs <https://sphinx-ru.readthedocs.io/ru/latest/rst-markup.html>`_

Resources
---------
* `Budges for README.rst <https://github.com/Naereen/badges/blob/master/README.rst>`_


Bugs
----

Please report new bugs via the `Github issue tracker <https://github.com/rorni/mckit/issues>`_.


Development
-----------

For Linux we assume usage of pyenv/poetry toolchain.

.. TODO explain details for activation of development environment
.. TODO add MKL handling stuff.
