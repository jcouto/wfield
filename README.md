# Widefield analysis tools

This is a python package for visualizing and analysing data collected with a widefield macroscope.

### What this can do:
  - Motion correction
  - Data reduction
  - Hemodynamic correction
  - Matching to the Allen CCF
  - Extract ROIs
  - Visualize raw/reduced data and extracted ROIs

### Command line interface

The command line interface can be used for pre-processing and data visualization.

To get a list of commands available do:

- `wfieldtools -h`

To preprocess WidefieldImager data do:

- `wfieldtools imager <DATAFOLDER> -o <LOCAL/DESTINATION FOLDER>`
- Complete example `wfieldtools imager \\\\grid-hs.cshl.edu\\churchland_nlsas_data\\data\\BpodImager\\Animals\\CSP23\\SpatialDisc\\12-Mar-2020 -o /c/data/CSP23/SpatialDisc/12-Mar-2020`


to list other options do:

- `wfieldtools imager -h`

To launch the GUI to explore processed data do:

- `wfieldtools open <FOLDER>`

### Example datasets

Example datasets are here.

### Installation

To install start by getting [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Install for all users and set as system python (this is not essential if you know what you are doing).

If you are using Windows, get a terminal like [git bash](https://git-scm.com/downloads) [optional]

- Go to the folder where you want to install and clone the repository: `git clone https://github.com/jcouto/wfield.git`. This create a directory; go inside the directory.

- Use anaconda to install all dependencies: `conda env create -f env.yml` the file env.yml is inside the wfield directory.

- Enter the environment `conda activate wfield` and onstall `wfieldtools` in develop mode so that local changes to the code take effect: `python setup.py develop` (you can not move the folder after this)

- You will need to run `conda activate wfield` to activate the environment before running the software.

### Tutorial

Tutorials are [here](https://github.com/jcouto/wfield/tree/master/notebooks).