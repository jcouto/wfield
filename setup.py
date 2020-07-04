#!/usr/bin/env python
# Install script for widefield tools

#  wfield - tools to analyse widefield data
# Copyright (C) 2020 Joao Couto - jpcouto@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
            'wfield-ncaas = wfield.ncaas_gui:main',
        ]
    },
    data_files=[(wfield_dir, [r]) for r in reference_files],
)
