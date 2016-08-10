from __future__ import print_function

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
import sys

if sys.version_info < (2, 7):
    print('You must have at least Python 2.7 for pyOZ to work correctly.\n',
          file=sys.stderr)
    sys.exit(0)

# pyoz package and all its subpackages
packages = ['pyoz', 'pyoz.propconv']

if __name__ == '__main__':
    import pyoz
    setup(name='pyoz',
          version=pyoz.__version__,
          description=pyoz.__doc__,
          author=pyoz.__author__,
          author_email='pyoz@vrbka.net',
          url='http://pyoz.vrbka.net/',
          license='BSD',
          packages=packages,
          install_requires=['scipy', 'simtk.unit'],
    )