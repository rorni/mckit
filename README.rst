.. include preamble.rst

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
.. TODO add pyenv/poetry stuff.

.. code-block:: sh
    git clone git@github.com:rorni/mckit.git
    cd mckit
    python setup.py build
    python setup.py install


Tutorial
--------
See the notebook in ``tutorial`` folder.

References
----------
* `Build and dependencies management <https://cjolowicz.github.io/posts/hypermodern-python-01-setup>`_
* `Linting <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
* `Стандартный синтаксис разметки  reStructuredText <https://sphinx-ru.readthedocs.io/ru/latest/rst-markup.html>`_


Bugs
----

Please report new bugs via the `Github issue tracker <git@github.com:rorni/mckit.git >`_.


TODO
----
.. TODO add nearest future plans.

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
