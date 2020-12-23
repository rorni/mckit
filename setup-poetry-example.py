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
 'jupyterlab>=2.2.9,<3.0.0',
 'mkl-devel>=2021.1.1,<2022.0.0',
 'mkl-include>=2021.1.1,<2022.0.0',
 'mkl>=2021.1.1,<2022.0.0',
 'nlopt>=2.6.2,<3.0.0',
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
{':python_version < "3.8"': ['importlib-metadata>=2.0.0,<3.0.0']}

entry_points = \
{'console_scripts': ['mckit = mckit.cli.runner:mckit']}

setup_kwargs = {
    'name': 'mckit',
    'version': '0.5.1a0',
    'description': 'Tools to process MCNP models and results',
    'long_description': None,
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
