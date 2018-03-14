from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
   Extension("box", ["box.pyx"],
       include_dirs = ["$GSL_INC"],
       library_dirs = ["$GSL_LIB"]
   )
]

setup(
   ext_modules = cythonize(extensions)
)