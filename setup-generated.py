# -*- coding: utf-8 -*-
from setuptools import setup

packages = [
    "mckit",
    "mckit.cli",
    "mckit.cli.commands",
    "mckit.parser",
    "mckit.parser.common",
    "mckit.utils",
]

package_data = {"": ["*"], "mckit": ["data/*", "src/*"]}

install_requires = [
    "DateTime>=4.3,<5.0",
    "Pygments>=2.7.2,<3.0.0",
    "aiofile>=3.3.3,<4.0.0",
    "asyncio>=3.4.3,<4.0.0",
    "atomicwrites>=1.4.0,<2.0.0",
    "attrs>=20.2.0,<21.0.0",
    "click-loguru>=1.3.4,<2.0.0",
    "click>=7.1.2,<8.0.0",
    "colorama>=0.4.4,<0.5.0",
    "isort>=5.7.0,<6.0.0",
    "mkl-devel>=2021.1.1,<2022.0.0",
    "mkl-include>=2021.1.1,<2022.0.0",
    "mkl>=2021.1.1,<2022.0.0",
    "numpy>=1.19.4,<2.0.0",
    "numpydoc>=1.1.0,<2.0.0",
    "ply>=3.11,<4.0",
    "python-dotenv>=0.15.0,<0.16.0",
    "scipy>=1.5.3,<2.0.0",
    "sly>=0.4,<0.5",
    "tomlkit>=0.7.0,<0.8.0",
    "tqdm>=4.53.0,<5.0.0",
]

extras_require = {
    ':python_version < "3.8"': ["importlib-metadata>=2.0.0,<3.0.0"],
    ':python_version < "3.8" and sys_platform == "Windows"': ["certifi==2020.6.20"],
}

entry_points = {"console_scripts": ["mckit = mckit.cli.runner:mckit"]}

setup_kwargs = {
    "name": "mckit",
    "version": "0.5.1a0",
    "description": "Tools to process MCNP models and results",
    "long_description": '.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg\n   :target: https://github.com/dvp2015/mckit/graphs/commit-activity\n\n.. image:: https://github.com/dvp2015/mckit/workflows/Tests/badge.svg\n   :target: https://github.com/dvp2015/mckit/actions?workflow=Tests\n\n.. image:: https://codecov.io/gh/dvp2015/mckit/branch/devel/graph/badge.svg?token=05OFBQS3RX\n   :target: https://codecov.io/gh/dvp2015/mckit\n\n.. image:: https://img.shields.io/badge/code%20style-black-000000.svg\n   :target: https://github.com/psf/black\n\n.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336\n    :target: https://pycqa.github.io/isort/\n\n.. image:: https://img.shields.io/github/license/dvp2015/mckit\n   :target: https://github.com/dvp2015/mckit\n\n\n\nMCKIT: MCNP model and results processing utilities\n==================================================\n\n\n.. TODO The complete documentation is available in the following languages:\n\n* `English documentation`_\n* `Russian documentation`_\n\n.. _English documentation: https://mckit.readthedocs.io/en/latest/\n.. _Russian documentation: https://mckit.readthedocs.io/ru/latest/\n\n.. contents:: Table of contents\n\nUsage\n-----\n\nCommand line interface\n~~~~~~~~~~~~~~~~~~~~~~\n\n.. code-block:: bash\n\n    Usage: mckit [OPTIONS] COMMAND [ARGS]...\n\n      Tools to process MCNP models and results\n\n    Options:\n      --override / --no-override\n      --version                   Show the version and exit.\n      -v, --verbose               Log debugging info to stderr.  [default: False]\n      -q, --quiet                 Suppress info to stderr.  [default: False]\n      --logfile / --no-logfile    Log to file.  [default: True]\n      --profile_mem               Profile peak memory use.  [default: False]\n      --help                      Show this message and exit.\n\n    Commands:\n      check      Read MCNP model(s) and show statistics and clashes.\n      compose    Merge universes and envelopes into MCNP model using merge...\n      concat     Concat text files.\n      decompose  Separate an MCNP model to envelopes and filling universes\n      split      Splits MCNP model to text portions (opposite to concat)\n      transform  Transform MCNP model(s) with one of specified transformation.\n\n\nLibrary\n~~~~~~~\n\nThe library allows subtraction and merging models, renaming objects (cells, surfaces, compositions, universes),\nsimplification of cell expressions (removing redundant surfaces), homogenization, computation of cell volumes and\nmaterial masses, and more.\n\n.. code-block:: python\n\n    LOG.info("Loading c-model envelopes")\n    envelopes = load_model(str(CMODEL_ROOT / "c-model.universes/envelopes.i"))\n\n    cells_to_fill = [11, 14, 75]\n    cells_to_fill_indexes = [c - 1 for c in cells_to_fill]\n\n    LOG.info("Attaching bounding boxes to c-model envelopes %s", cells_to_fill)\n    attach_bounding_boxes(\n        [envelopes[i] for i in cells_to_fill_indexes], tolerance=5.0, chunk_size=1\n    )\n    LOG.info("Backing up original envelopes")\n    envelopes_original = envelopes.copy()\n\n    antenna_envelop.rename(start_cell=200000, start_surf=200000)\n\n    LOG.info("Subtracting antenna envelop from c-model envelopes %s", cells_to_fill)\n    envelopes = subtract_model_from_model(\n        envelopes, antenna_envelop, cells_filter=lambda c: c in cells_to_fill\n    )\n    LOG.info("Adding antenna envelop to c-model envelopes")\n    envelopes.add_cells(antenna_envelop, name_rule="clash")\n    envelopes_path = "envelopes+antenna-envelop.i"\n    envelopes.save(envelopes_path)\n    LOG.info("The model of HFSR in envelopes is saved to %s", envelopes_path)\n\n\n\nInstallation\n------------\n\nInstalling from pypi:\n\n.. code-block:: bash\n\n    pip3 install mckit\n\n\nInstalling from github.com:\n\n.. code-block:: bash\n\n    pip3 install git+https://github.com/MC-kit/mckit.git\n\n\nVersioning\n----------\n\nThis software follows `Semantic Versioning`_\n\n.. _Semantic Versioning: http://semver.org/\n\n\nContributors\n------------\n\n* `Roman Rodionov <mailto:r.rodionov@iterrf.ru>`_\n* `Dmitri Portnov <mailto:dmitri_portnov@yahoo.com>`_\n',
    "author": "rrn",
    "author_email": "r.rodionov@iterrf.ru",
    "maintainer": "dpv2015",
    "maintainer_email": "dmitri_portnov@yahoo.com",
    "url": "https://github.com/rorni/mckit",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "extras_require": extras_require,
    "entry_points": entry_points,
    "python_requires": ">=3.7,<4.0",
}
from build import *

build(setup_kwargs)

setup(**setup_kwargs)
