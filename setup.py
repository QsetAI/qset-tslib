from setuptools import find_packages

# cython solution from here: http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
# use "python setup.py build_ext --inplace" command to build locally.
from distutils.core import setup
from distutils.extension import Extension

from distutils.command.sdist import sdist as _sdist

import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []


class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(['qset_tslib/cython/neutralize/cneutralize.pyx'])
        cythonize(['qset_tslib/cpp/ts/cts.pyx'])
        _sdist.run(self)


cmdclass['sdist'] = sdist

if use_cython:
    ext_modules += [
        Extension('qset_tslib.cython.neutralize.cneutralize', ['qset_tslib/cython/neutralize/cneutralize.pyx'], language='c++'),
        Extension('qset_tslib.cpp.ts.cts', ['qset_tslib/cpp/ts/cts.pyx', 'qset_tslib/cpp/ts/ts.cpp'], language='c++')
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension('qset_tslib.cython.neutralize.cneutralize', ["qset_tslib/cpp/neutralize/cneutralize.cpp"], language='c++'),
        Extension('qset_tslib.cpp.ts.cts', ['qset_tslib/cpp/ts/ts.cpp'], language='c++')
    ]

setup(name='qset_tslib',
      version='1.0.0',
      description='Qset Fintech Utils Library',
      packages=find_packages(),
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      install_requires=['joblib', 'h5py', 'empyrical', 'anyconfig'],
      include_dirs=[np.get_include()],
      zip_safe=False)
