from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import sys

nlopt_inc = "C:\\ProgramData\\Libs\\nlopt"
nlopt_lib = "C:\\ProgramData\\Libs\\nlopt"
mkl_inc = sys.prefix + '\\Library\\include'
mkl_lib = sys.prefix + '\\Library\\lib'

extensions = [
   Extension("geometry", ["src/geometrymodule.c", "src/box.c", "src/surface.c", "src/shape.c", "src/rbtree.c"],
       include_dirs = [np.get_include(), mkl_inc, nlopt_inc],
       libraries = ['mkl_intel_lp64_dll', 'mkl_core_dll', 
                    'mkl_sequential_dll', 'libnlopt-0'],
       library_dirs = [mkl_lib, nlopt_lib],
   )
]

setup(
   ext_modules = extensions
)