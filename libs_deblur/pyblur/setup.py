from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pyblur',
      version = '0.2.3',
      description = 'Image blurring routines',
      long_description = long_description,
      keywords = 'blur',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2.7',
          'Topic :: Multimedia :: Graphics'],
      url='http://github.com/lospooky/pyblur',
      author='lospooky',
      author_email='my.accounts@gmx.se',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires = ['numpy', 'pillow', 'scikit-image', 'scipy'],
      zip_safe = False)