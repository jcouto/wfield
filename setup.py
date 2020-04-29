#!/usr/bin/env python
# Install script for widefield tools
# Joao Couto - March 2020
# jpcouto@gmail.com

from setuptools import setup
from setuptools.command.install import install
import os
from os.path import join as pjoin

wfield_dir = pjoin(os.path.expanduser('~'),'.wfield')
if not os.path.isdir(wfield_dir):
    print('Creating {0}'.format(wfield_dir))
    os.makedirs(wfield_dir)

reference_files =  [os.path.abspath(pjoin('references',f))
                    for f in os.listdir('references')
                    if os.path.isfile(pjoin('references', f))]

longdescription = '''Utilities to look at widefield data and align with the allen reference map.'''
setup(
    name = 'wfield',
    version = '0.0',
    author = 'Joao Couto',
    author_email = 'jpcouto@gmail.com',
    description = (longdescription),
    long_description = longdescription,
    license = 'GPL',
    packages = ['wfield'],
    entry_points = {
        'console_scripts': [
            'wfield = wfield.cli:main',
        ]
    },
    data_files=[(wfield_dir, [r]) for r in reference_files],
)
