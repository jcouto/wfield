import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import time
from natsort import natsorted
from glob import glob
from os.path import join as pjoin
from scipy.signal import medfilt
from multiprocessing import Pool,cpu_count
from functools import partial

def analog_ttl_to_onsets(dat,time=None, mfilt=3):
    if time is None:
        time = np.arange(len(dat))
    if medfilt:
        dat = medfilt(dat,mfilt)
    tt = np.diff(dat.astype(np.float32))
    onsets = np.where(tt-np.max(tt)/2 > 0)[0]+1
    return time[onsets]


def make_overlapping_blocks(dims,blocksize=128,overlap=16):
    '''
    Creates overlapping block indices to span an image
    '''
    
    w,h=dims
    blocks = []
    for i,a in enumerate(range(0,w,blocksize-overlap)):
        for j,b in enumerate(range(0,h,blocksize-overlap)):
            blocks.append([(a,np.clip(a+blocksize,0,w)),(b,np.clip(b+blocksize,0,h))])
    return blocks

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

def runpar(f,X,nprocesses = None,**kwargs):
    ''' 
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)

    '''
    if nprocesses is None:
        nprocesses = cpu_count()
    with Pool(initializer = parinit, processes=nprocesses) as pool:
        res = pool.map(partial(f,**kwargs),X)
    pool.join()
    return res
