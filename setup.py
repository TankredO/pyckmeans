''' setup
'''

from distutils.command.build_ext import build_ext as build_ext_orig
from setuptools import setup, find_packages, Extension

# ==== ctypes extensions
class CTypesExtension(Extension):
    '''CTypesExtension'''

class build_ext(build_ext_orig):
    '''build_ext'''
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)

distance_module = CTypesExtension(
    'ckmeans.distance.lib.distance',
    sources=['ckmeans/distance/src/distance.cpp'],
)

ext_modules = [distance_module]

# ====
setup(
    name='ckmeans',
    packages=find_packages(),
    description='A consensus K-Means implementation.',
    author='Tankred Ott',
    platforms=['any'],
    python_requires='>=3.6',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
