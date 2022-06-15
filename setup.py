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
from distutils.cmd import Command

description = '''Utilities to look at widefield data and align with the allen reference map.'''

with open("README.md", "r") as fh:
    longdescription = fh.read()

requirements = []
with open("requirements.txt","r") as f:
    requirements = f.read().splitlines()

class AddReferences(Command):
    """ """
    
    description = 'create the wfield folder and add reference maps and files.'
    user_options = []
    
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        wfield_dir = pjoin(os.path.expanduser('~'),'.wfield')
        if not os.path.isdir(wfield_dir):
            os.makedirs(wfield_dir)
        refpath = 'references'
        reference_files = [pjoin(refpath,r) for r in os.listdir(refpath)]
        from shutil import copyfile
        for f in reference_files:
            if os.path.isfile(f):
                copyfile(f,f.replace(refpath,wfield_dir))
                print('{0} copied to {1}'.format(f,wfield_dir))

    
setup(
    name = 'wfield',
    version = '0.3.1',
    author = 'Joao Couto',
    author_email = 'jpcouto@gmail.com',
    description = (description),
    long_description = longdescription,
    long_description_content_type='text/markdown',
    license = 'GPL',
    install_requires = requirements,
    url = "https://github.com/jcouto/wfield",
    packages = ['wfield'],
    cmdclass = {'references' : AddReferences},
    include_package_data = True,
    package_data = {'share/wfield/references':['references/*.json','references/*.npy']},
    #data_files = [('share/wfield/references',['references/'+f]) for f in os.listdir('references')],
    entry_points = {
        'console_scripts': [
            'wfield = wfield.cli:main',
            'wfield-ncaas = wfield.ncaas_gui:main',
        ],
        
    })
