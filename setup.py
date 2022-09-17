from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name="pure_cython", ext_modules=cythonize('pure_cython.pyx'), include_dirs=[np.get_include()])

# run in terminal:
# 1) cd c_functions
# 2) python setup.py build_ext --inplace
