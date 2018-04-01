from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import sys

nlopt_inc = "C:\\Libs\\nlopt\\include"
nlopt_lib = "C:\\Libs\\nlopt\\lib"
mkl_inc = sys.prefix + '\\Library\\include'
mkl_lib = sys.prefix + '\\Library\\lib'

extensions = [
   Extension("geomext", ["wrap/geomext.pyx", "src/box.c",
                         "src/surface.c"],
       include_dirs = [np.get_include(), mkl_inc, nlopt_inc],
       libraries = ['mkl_intel_lp64_dll', 'mkl_core_dll', 
                    'mkl_sequential_dll', 'libnlopt-0'],
       library_dirs = [mkl_lib, nlopt_lib],
   )
]

setup(
   ext_modules = cythonize(extensions)
)