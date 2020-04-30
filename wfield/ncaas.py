# This has wrappers and utilities for running or vizualizing data on NeuroCAAS
# Link, references and names should go here.
#
# This has functions to:
#     - Interface with aws to upload data to the cloud
#     - Functions to get experiments from the cloud
#     - Wrappers for data viz and run some parts
#
# Wrapper to run motion correction that takes a filename and kicks out a structure (first step of 'this' blueprint - link)
# 
from .utils import *
from .io import load_dat, mmap_dat
from .registration import motion_correct

def load_and_motion_correct(filename,
                            outputfolder = None,
                            chunksize=1048,
                            nreference = 60,
                            mmap = False,
                            flatten_frames=False,
                            outputdir = None):
    '''
    Motion correction by translation.
    This estimates x and y shifts using phase correlation. And applies shifts to the dataset.
    
    The reference image is the average (nreference frames) from the chunk in the center.

Inputs:
    filename              : the name of the file to perform correction on a data file
    chunksize (int)       : size of the chunks where fft is performed (default 512)
    nreference            : number of frames to take as reference (default 60)
    mmap                  : default False; if True, result is written to disk and returns a memory mapped array
    flatten_frames        : returns [NFRAMES * NCHANNELS, H, W] instead
    outputdir             : folder to write the results to.

Returns:
    motion corrected frames [NFRAMES, NCHANNELS, H, W]
    '''

    if mmap:
        # motion corrected frames overwrite the raw data
        dat = mmap_dat(filename, mode='r+')
    else:
        # this will take a while, loads the entire dataset to memory
        dat = load_dat(filename) 
        
    yshifts,xshifts = motion_correct(dat, chunksize=chunksize,
                                     nreference = nreference,
                                     apply_shifts= True)

    if outputdir is None:
        outputdir = os.path.dirname(filename)
        # save the shifts
        shifts = np.rec.array([yshifts,xshifts],dtype=[('y','int'),('x','int')])
        np.save(pjoin(outputdir,'motion_correction_shifts.npy'),shifts)

        try: # don't crash if there are issues plotting
            from .plots import plot_summary_motion_correction 
            import pylab as plt
            plt.matplotlib.style.use('ggplot')
            plot_summary_motion_correction(shifts,localdisk = outputdir);
        except Exception as err:
            print('There was an issue plotting.')
            print(err)

    if mmap:
        del dat
        dat = mmap_dat(dat_path, mode='r')
        
    if flatten_frames:
        return dat.reshape([-1,*dat.shape[-2:]])

    return dat


