.. include:: preamble.rst

MCKIT
=====

Introduction
------------

The package defines console executable |mckit| and library to work with
MCNP models and results. The program |mckit| allows compose and decompose a MCNP
model over universes hierarchy, split a model to text portions and combine
back.

.. tip::

   To see actual list of available commands run::

       mckit --help

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


Authors
-------
* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_

Maintainers
-----------
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

TODO
~~~~

* implement generation of the documentation for readthedocs
* create and link to the readocs account
* add the readthedocs status icon at the document start.

References
----------
* `Build and dependencies management <https://cjolowicz.github.io/posts/hypermodern-python-01-setup>`_
* `Managing Multiple Python Versions With pyenv <https://realpython.com/intro-to-pyenv/>`_
* `Linting <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
* `Стандартный синтаксис разметки  reStructuredText <https://sphinx-ru.readthedocs.io/ru/latest/rst-markup.html>`_
* `Pytest <https://docs.pytest.org/en/stable/index.html>`_
* `Python development best practices 2019 <https://flynn.gg/blog/software-best-practices-python-2019/>`_
* `Poetry  <https://python-poetry.org/>`_

Bugs
----

Please report new bugs via the `Github issue tracker <https://github.com/rorni/mckit/issues>`_.


TODO
----
.. TODO add nearest future plans.


Development
-----------

For Linux we assume usage of pyenv/poetry toolchain.

* TODO explain details for activation of development environment
* TODO add MKL handling stuff.
