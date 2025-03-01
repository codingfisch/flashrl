from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

compile_args=['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-O0']
filepaths = ['flashrl/envs/grid/cy_grid.pyx']
ext_mods = [Extension(fp.replace('/', '.')[:-4], sources=[fp], extra_compile_args=compile_args) for fp in filepaths]
setup(name='flashrl', ext_modules=cythonize(ext_mods), include_dirs=[numpy.get_include()])
