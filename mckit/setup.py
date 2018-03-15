from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

gsl_inc = os.path.expandvars("$GSL_INC")
gsl_lib = os.path.expandvars("$GSL_LIB")

extensions = [
   Extension("box", ["wrap/box.pyx", "src/box.c"],
       include_dirs = [gsl_inc, np.get_include()],
       libraries = ['libgslcblas'],
       library_dirs = [gsl_lib]
   )
]

setup(
   ext_modules = cythonize(extensions)
)