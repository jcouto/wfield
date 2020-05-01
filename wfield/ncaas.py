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
from .hemocorrection import hemodynamic_correction

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


def dual_color_hemodymamic_correction(U,SVT, output_folder = None,
                                      frame_rate = 30.,
                                      freq_lowpass = 15., 
                                      freq_highpass = 0.1):
    ''' 
    Hemodynamic correction on dual channel data.
    The blue channel is assumed to be the first.
    
    Inputs:
    
    U (array)     : spatial components (H,W,NCOMPONENTS)
    SVT (array)   : temporal components (NCOMPONENTS,NFRAMES*2CHANNELS)
    output_folder : where to store results and figures (default current directory)
    frame_rate    : frame rate of the acquisition 
    freq_lowpass  : lowpass frequency (only for violet - default 15. (none at 30Hz))
    freq_highpass : frequency of the highpass filter, applied to both channels (default is 0.1)

    Returns:
    SVTcorr      : the corrected temporal components (NCOMPONENTS, NFRAMES)
    '''
    
    if output_folder is None:
        output_folder = os.path.abspath(os.path.curdir)
        print('Output not specified, using {0}'.format(output_folder))
    
    SVTcorr, rcoeffs, T = hemodynamic_correction(U, SVT, 
                                                 fs=frame_rate,
                                                 freq_lowpass=freq_lowpass,
                                                 freq_highpass = freq_highpass)        

    np.save(pjoin(output_folder,'rcoeffs.npy'),rcoeffs) # regression coefficients
    np.save(pjoin(output_folder,'T.npy'),T)             # transformation matrix
    np.save(pjoin(output_folder,'SVTcorr.npy'),SVTcorr) # corrected SVT

    try: # don't crash while plotting
        import pylab as plt
        plt.matplotlib.style.use('ggplot')
        from wfield import  plot_summary_hemodynamics_dual_colors
        plot_summary_hemodynamics_dual_colors(rcoeffs,
                                              SVT,
                                              U,
                                              T,
                                              frame_rate=frame_rate,
                                              duration_frames = 60,
                                              outputdir = output_folder);
    except Exception as err:
        print('There was an issue plotting.')
        print(err)
        
    return SVTcorr

