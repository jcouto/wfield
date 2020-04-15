#!/usr/bin/env python
# Install script for widefield tools
# Joao Couto - March 2020
# jpcouto@gmail.com

from setuptools import setup
from setuptools.command.install import install


longdescription = '''Utilities to look at widefield data and align with the allen reference map.'''
setup(
    name = 'wfieldtools',
    version = '0.0',
    author = 'Joao Couto',
    author_email = 'jpcouto@gmail.com',
    description = (longdescription),
    long_description = longdescription,
    license = 'GPL',
    packages = ['wfieldtools'],
    entry_points = {
        'console_scripts': [
            'wfieldtools = wfieldtools.cli:main',
        ]
    },
)
