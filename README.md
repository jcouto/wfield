# Tools to analyze widefield data 

This is a python package for visualizing and analysing data collected with a widefield macroscope.

### What this can do:
  - Motion correction
  - Data reduction
  - Hemodynamic correction
  - Matching to the Allen CCF
  - Extract ROIs
  - Visualize raw/reduced data and extracted ROIs

### Use cases and instructions [here](usecases.md)

A [dataset](http://repository.cshl.edu/id/eprint/38599/) that can be used to demo some of the functionality of this repository is made available in the ``demoRec`` folder. Follow to instructions in the GUI for demo in NeuroCAAS.

### File format conventions

  - raw frame data is stored in binary files (uint16) <br />
    The filename must end with: `_NCHANNELS_H_W_DTYPE.dat` <br />
    Example: "frames_2_540_640_uint16.dat" H and W are the dimensions of a single frame. <br /> 
  
  - denoised/decomposed data are stored as `npy` arrays <br /> 
  `U.npy` are the spatial components `(H, W, NCOMPONENTS)` <br />
  `SVT.npy` are the temporal components `(NCOMPONENTS, NFRAMES)` <br />
  `sparse_spatial.npz` are the spatial components from denoised PMD `(H*W,NCOMPONENTS)` these are sparse matrices stored in compressed format. <br />
  
  - `VSTcorr.npy` is the hemodynamic corrected temporal components `(NCOMPONENTS, NFRAMES)`
  
  - `info.json` has information about the dataset like the `frame_rate` or the `n_channels`
    
### Installation

To install start by getting [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Go to the folder where you want to install and clone the repository: ``git clone https://github.com/jcouto/wfield.git``. This creates a directory; go inside that directory: ``cd wfield``.

2. Use anaconda to install all dependencies: ``conda env create -f env.yml`` the file env.yml is inside the ``wfield`` directory.

3. Enter the environment ``conda activate wfield`` and install wfield using the command ``python setup.py install``

4. You will need to run `conda activate wfield` to activate the environment before running the software every time you start a terminal.

5. You are all set. <br /> Type ``wfield -h`` to see the available commands. <br /> Go here for [instructions](https://github.com/jcouto/wfield/tree/master/usecases.md) on how to use NeuroCAAS.


*Note:* Some reference files used to match to the Allen Common Coordinate Framework are copied from the folder [references](https://github.com/jcouto/wfield/tree/master/references) to ``$HOME/.wfield`` during installation. 

*Note for Mac users:*

   - ``git`` when you try the instructions you will be asked to install git, if that fails you can run ``conda install git`` to install using anaconda. 

*Note for Windows users:*

   - Get a terminal like [git bash](https://git-scm.com/downloads) [optional] <br />
   Run ``conda init bash`` to activate conda on ``git bash``
   - When you install Anaconda,  set the option to install as system python (this makes that it is visible from the terminal without having to run the Anaconda Prompt).


*Note for developers:* In some cases you may want to make changes to the software, if you need this run ``python setup.py develop`` (you can not move the folder after this - the installation will point to that directory).


The software was tested on Windows, Linux and MacOS Catalina. Installation takes less than 5 minutes on a standard computer with fast access to internet and a previous anaconda installation.


### Tutorial

Instructions to use with NeuroCAAS [here](https://github.com/jcouto/wfield/tree/master/usecases.md).


Tutorials are [here](https://github.com/jcouto/wfield/tree/master/notebooks).



Copyright (C) 2020 Joao Couto - jpcouto@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
