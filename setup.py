from setuptools import setup
from setuptools import find_packages

import re
import os
import sys
from pkg_resources import parse_version

# Dependencies of PILCO
requirements = [
    'gpflow>=1.0,<2.0'
]

packages = find_packages('.')
setup(name='pilco',
      version='0.1',
      author="Nikitas Rontsis, Kyriakos Polymenakos",
      author_email="nrontsis@gmail.com",
      description=("A TensorFlow implementation of PILCO"),
      license="MIT License",
      keywords="reinforcement-learning model-based-rl gaussian-processes tensorflow machine-learning",
      url="http://github.com/nrontsis/PILCO",
      packages=packages,
      install_requires=requirements,
      include_package_data=True,
      test_suite='tests',
      extras_require={'Tensorflow with GPU': ['tensorflow-gpu>=1.5']},
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
