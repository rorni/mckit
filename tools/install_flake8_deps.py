#!/usr/bin/env python
"""Install selected flake8 plugins to current poetry environment.

Descriptions
------------
flake8-builtins - make sure you don’t accidentally name a variable the same thing as a builtin. This happens a lot with id.
flake8-bugbear - “find likely bugs and design problems in your program”, like when you have an unused variable in a loop
flake8-colors - add color to the flake8 output (explanation how to set up is below)
flake8-commas - add trailing commas where appropriate
flake8-comprehensions reminders to use list comprehensions where appropriate
flake8-docstrings - make sure your docstrings are present and written in the right format
flake8-import-order - make sure your imports are organized properly
flake8-print - make sure you never ever use print(). The literal only exception is when using print to get text into a file with print(..., file=...)
flake8-use-fstring - make sure you’re using f-strings instead of % or .format() formatting. Exception being for logging.
pep8-naming - make sure names of variables, classes, and modules look right.
pydocstyle - docstring style checker

References
----------
Partially borrowed from:

- https://cthoyt.com/2020/04/25/how-to-code-with-me-flake8.html

"""  # noqa: ignore B950

import subprocess

deps = [
    "flake8",
    "flake8-annotations",
    # TODO dvp: versions 3.0.0 and older don't work with recent flake8, check on update
    #  "flake8-bandit",
    "flake8-bugbear",
    "flake8-builtins",
    "flake8-colors",
    "flake8-commas",
    "flake8-comprehensions",
    "flake8-docstrings",
    "flake8-import-order",
    "flake8-print",
    "flake8-rst-docstrings",
    "flake8-use-fstring",
    "pep8-naming",
    "pydocstyle",
    "tryceratops",
]

subprocess.run(["poetry", "add", "--dev", *deps])
