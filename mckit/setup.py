from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import sys

mkl_inc = sys.prefix + '\\Library\\include'
mkl_lib = sys.prefix + '\\Library\\lib'

extensions = [
   Extension("box", ["wrap/box.pyx", "src/box.c"],
       include_dirs = [np.get_include(), mkl_inc],
       libraries = ['mkl_intel_lp64_dll', 'mkl_core_dll', 'mkl_sequential_dll'],
       library_dirs = [mkl_lib],
   )
]

setup(
   ext_modules = cythonize(extensions)
)