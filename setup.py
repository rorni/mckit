from setuptools import setup, find_packages

setup(
    name='mckit',
    version='0.1',
    packages=find_packages(),
    package_data={'mckit': ['data/isotopes.dat']},
    url='https://gitlab.iterrf.ru/Rodionov/mckit',
    license='',
    author='Roman Rodionov',
    author_email='r.rodionov@iterrf.ru',
    description='Tool for handling neutronic models and results',
    install_requires=['numpy', 'scipy']
)
