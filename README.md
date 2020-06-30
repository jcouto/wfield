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

### File format conventions

  - raw frame data is stored in binary files (uint16) <br />
    The filename must end with: `_NCHANNELS_H_W_DTYPE.dat` <br />
    Example: "frames_2_540_640_uint16.dat" H and W are the dimensions of a single frame. <br /> 
  
  - denoised/decomposed data are stored as `npy` arrays <br /> 
  `U.npy` are the spatial components `(H, W, NCOMPONENTS)` <br />
  `VST.npy` are the temporal components `(NCOMPONENTS, NFRAMES)` <br /> 
  
  - `VSTcorr.npy` is the hemodynamic corrected temporal components `(NCOMPONENTS, NFRAMES)`
  
  - `info.json` has information about the dataset like the `frame_rate` or the `n_channels`
    
### Installation

To install start by getting [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Install for all users and set as system python (this is not essential if you know what you are doing). 

If you are using Windows, get a terminal like [git bash](https://git-scm.com/downloads) [optional] Run ``conda init bash``

- Go to the folder where you want to install and clone the repository: ``git clone https://github.com/jcouto/wfield.git``. This creates a directory; go inside that directory: ``cd wfield``.

- Use anaconda to install all dependencies: ``conda env create -f env.yml`` the file env.yml is inside the ``wfield`` directory.

- Enter the environment ``conda activate wfield`` and install wfield using the command ``python setup.py install``


*Note:* You will need to run `conda activate wfield` to activate the environment before running the software every time you start a terminal.

*Note:* In some cases you may want to make changes to the software, if you need this run ``python setup.py develop`` (you can not move the folder after this - the installation will point to that directory).

*Note:* Some reference files used to match to the Allen Common Coordinate Framework are copied from the folder [references](https://github.com/jcouto/wfield/tree/master/references) to ``$HOME/.wfield`` during installation. 



### Tutorial

Instructions to use with NeuroCAAS [here](https://github.com/jcouto/wfield/tree/master/usecases.md).


Tutorials are [here](https://github.com/jcouto/wfield/tree/master/notebooks).


