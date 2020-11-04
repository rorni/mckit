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

.. code:: sh
    git clone git@github.com:rorni/mckit.git
    cd mckit
    python setup.py build
    python setup.py install

Now break it and tell me about it (or use it smoothly and tell me about that, too).


Tutorial
--------

See the notebook in ``tutorial`` folder.

References
----------

* `Build and dependencies management <https://cjolowicz.github.io/posts/hypermodern-python-01-setup>`_
* `Linting <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`_
* `Стандартный синтаксис разметки  reStructuredText <https://sphinx-ru.readthedocs.io/ru/latest/rst-markup.html>`_

## DOCS

Consult the docstrings for module, class, and function documentation.

```sh
python -c 'import cdb; print cdb.__doc__'
python -c 'import cdb; print cdb.cdbmake("f.cdb","f.tmp").__doc__'
python -c 'import cdb; print cdb.init("some.cdb").__doc__'
```


## BUGS

Please report new bugs via the [Github issue tracker](https://github.com/acg/python-cdb/issues).


## TODO

- [ ] more dict-like API
- [ ] test cases
- [ ] take advantage of contemporary Python API
- [ ] formal speed benchmarks
- [ ] possibly revert to DJB's cdb implementation; explicitly public domain since 2007Q4
- [ ] better README/docs
- [ ] mingw support


## COPYRIGHT

`python-cdb` is free software, as is cdb itself.

The extension module is licensed under the GNU GPL version 2 or later, and is copyright 2001, 2002 Michael J. Pomraning.  Ancillary files from Felix von Leitner's libowfat are also licensed under the GPL.  Finally, modifications to D. J. Bernstein's public domain cdb implementation are similarly released to the public domain.


## AUTHORS

- Alan Grow <alangrow+python-cdb@gmail.com>
- Mike Pomraning <mjp@pilcrow.madison.wi.us>

