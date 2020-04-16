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
from scipy.signal import butter,filtfilt

print = partial(print, flush=True)

def lowpass(X, w = 7.5, fs = 30.):
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=50)

def highpass(X, w = 3., fs = 30.):
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=50)


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

def reconstruct(u,svt,dims):
    return np.dot(u,svt).reshape((*dims,-1)).transpose(-1,0,1)


class SVDStack(object):
    def __init__(self,U,SVT,dims,dtype = 'float32'):
        self.U = U.astype('float32')
        self.SVT = SVT.astype('float32')
        self.shape = [SVT.shape[1],*dims]
        self.dtype = dtype
    def __len__(self):
        return self.SVT.shape[1]
    def __getitem__(self,*args):
        ndims  = len(args)
        if type(args[0]) is slice:
            idxz = range(*args[0].indices(self.shape[0]))
        else:
            idxz = args[0]        
        return reconstruct(self.U,self.SVT[:,idxz],self.shape[1:]).squeeze()
    def get_timecourse(xy):
        # TODO: this needs a better interface
        x = int(np.clip(xy[0],0,self.shape[1]))
        y = int(np.clip(yy[1],0,self.shape[2]))
        idx = np.ravel_multi_index((x,y),self.shape[1:])
        t = np.dot(self.U[idx,:],self.SVT)
        return t

def get_trial_baseline(idx,frames_average,onsets):
    if len(frames_average.shape) <= 3:
        return frames_average
    else:
        if onsets is None:
            print(' Trial onsets not defined, using the first trial')
            return frames_average[0]
        return frames_average[np.where(onsets<=idx)[0][-1]]

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
