#  wfield - tools to analyse widefield data - utils and general imports 
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
import os
import sys
import time

try:
    import cv2 # OpenCV needs to be imported before numpy for some seg faulted reason...
except:
    print('Some functionality might be broken: install opencv-python or opencv-python-headless')
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from os.path import join as pjoin
from datetime import datetime
from skimage.transform import warp
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.interpolate import interp1d
from scipy.sparse import load_npz, issparse,csr_matrix

print = partial(print, flush=True)

def estimate_similarity_transform(ref,points):
    '''
    
    ref = np.vstack([landmarks_im['x'],landmarks_im['y']]).T
    match = point_stream.data    
    cor = np.vstack([match['x'],match['y']]).T
    
    M = estimate_similarity_transform(ref, cor)
    '''
    from skimage.transform import SimilarityTransform
    M = SimilarityTransform()
    M.estimate(ref,points)
    return M 

def im_adapt_hist(im,clip_limit = .1, grid_size=(8,8)):
    ''' Adaptative histogram of image

        eqim = im_adapt_hist(im,.1)
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(im.squeeze())

def im_apply_transform(im,M,dims = None):
    '''
    Applies an affine transform M to an image.
    nim = im_apply_transform(im,M)

    '''
    if issparse(im):
        # then reshape before
        if dims is None:
            raise ValueError('Provide dims when warping sparse matrices.')
        shape = im.shape
        tmp  = np.asarray(im.todense()).reshape(dims)
        tmp = warp(tmp,M,
                   order = 1,
                   mode='constant',
                   cval = 0,
                   clip=True,
                   preserve_range=True)
        return csr_matrix(tmp.reshape(shape))
    else:    
        return warp(im,M,
                    order = 1,
                    mode='constant',
                    cval = 0,
                    clip=True,
                    preserve_range=True)

def lowpass(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=50)

def highpass(X, w = 3., fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=50)


def analog_ttl_to_onsets(dat,time=None, mfilt=3):
    if time is None:
        time = np.arange(len(dat))
    if mfilt:
        from scipy.signal import medfilt
        dat = medfilt(dat,mfilt)
    tt = np.diff(dat.astype(np.float32))
    onsets = np.where(tt-np.max(np.abs(tt))/2 > 0)[0]+1
    offsets = np.where(tt+np.max(np.abs(tt))/2 < 0)[0]+1
    return time[onsets],time[offsets]

def chunk_indices(nframes, chunksize = 512, min_chunk_size = 16):
    '''
    Gets chunk indices for iterating over an array in evenly sized chunks
    '''
    chunks = np.arange(0,nframes,chunksize,dtype = int)
    if (nframes - chunks[-1]) < min_chunk_size:
        chunks[-1] = nframes
    if not chunks[-1] == nframes:
        chunks = np.hstack([chunks,nframes])
    return [[chunks[i],chunks[i+1]] for i in range(len(chunks)-1)]


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

def reconstruct(u,svt,dims = None):
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]
    return u.dot(svt).reshape((*dims,-1)).transpose(-1,0,1).squeeze()


class SVDStack(object):
    def __init__(self, U, SVT, dims = None, warped = None, dtype = 'float32'):
        self.U = U.astype('float32')
        self.SVT = SVT.astype('float32')
        self.issparse = False
        if issparse(U):
            self.issparse = True
            if dims is None:
                raise ValueError('Supply dims = [H,W] when using sparse arrays')
            self.Uflat = self.U
        else:
            if dims is None:
                dims = U.shape[:2]
            self.Uflat = self.U.reshape(-1,self.U.shape[-1])
        self.U_warped = warped
        self.warped = False
        self.M = None
        self.shape = [SVT.shape[1],*dims]
        self.dtype = dtype
        self.originalU = None
    def set_warped(self,value,M = None):
        ''' Apply affine transform to the spatial components '''
        if not M is None:
            self.M = M
        if self.originalU is None:
            self.originalU = self.U.copy()
        if not value:
            self.U = self.originalU
            self.warped = False
        else:
            if self.U_warped is None:
                if not self.M is None:
                    if not self.issparse:
                        if self.originalU is None:
                            self.originalU = self.U.copy()
                        self.U_warped = self.originalU.copy()
                        self.U_warped[:,0,:] = 1e-10
                        self.U_warped[0,:,:] = 1e-10
                        self.U_warped[-1,:,:] = 1e-10
                        self.U_warped[:,-1,:] = 1e-10
                        self.U_warped = np.stack(runpar(im_apply_transform,
                                                        self.U_warped.transpose([2,0,1]),
                                                        M = self.M)).transpose([1,2,0])
            if not self.U_warped is None:
                self.U = self.U_warped
                self.warped = True
        self.Uflat = self.U.reshape(-1,self.U.shape[-1])
        return
    def __len__(self):
        return self.SVT.shape[1]
    def __getitem__(self,*args):
        ndims  = len(args)
        if type(args[0]) is slice:
            idxz = range(*args[0].indices(self.shape[0]))
        else:
            idxz = args[0]        
        return reconstruct(self.U,self.SVT[:,idxz],dims = self.shape[1:])
    def get_timecourse(self,xy):
        # index are in xy, like what np.where(mask) returns
        x = np.array(np.clip(xy[0],0,self.shape[1]),dtype=int)
        y = np.array(np.clip(xy[1],0,self.shape[2]),dtype=int)
        idx = np.ravel_multi_index((x,y),self.shape[1:])
        t = self.Uflat[idx,:].dot(self.SVT)
        return t

def get_trial_baseline(idx,frames_average,onsets):
    if len(frames_average.shape) <= 3:
        return frames_average
    else:
        if onsets is None:
            print(' Trial onsets not defined, using the first trial')
            return frames_average[0]
        return frames_average[np.where(onsets<=idx)[0][-1]]

def point_find_ccf_region(point,ccf_regions,sides = ['left','right'],approx_value=-0.01):
    '''
    Find the area where a point is contained from ccf_regions contours.
    Example:
    point = [6.5,3.0]
    region,side,index = point_find_ccf_region(point,refregions)
    
    '''
    region = None
    idx = None
    ir = None
    for ir, r in ccf_regions.iterrows():
        for s in sides:
            c = np.vstack([r[s + '_x'], r[s + '_y']]).astype('float32').T
            c = c.reshape([c.shape[0],1,c.shape[1]])
            d = cv2.pointPolygonTest(c,tuple(point),True)
            if d >= approx_value:
                region = r
                idx = ir
                side = s
        if not region is None:
            break
    return region,side,idx

def contour_to_im(x,y,dims,extent=None,n_up_samples = 1000):
    '''
    im = contour_to_im(x,y,dims,extent=None,n_up_samples = 1000)
    
    Imprint a contour on an image.
    
    see also: contour_to_mask
'''
    if extent is None:
        extent = [0,dims[0],0,dims[1]]
    
    cont = np.vstack([y,x]).T
    x = np.linspace(extent[0], extent[1], dims[0]+1)
    y = np.linspace(extent[2], extent[3], dims[1]+1)
    
    C = cont.copy()
    C = np.vstack([C,C[0,:]])
    if n_up_samples > cont.shape[0]:
        C = np.zeros((n_up_samples,2))
        C[:,0] = interp1d(np.linspace(0,1,cont.shape[0]),
                          np.clip(cont[:,0],0,dims[0]))(
                              np.linspace(0,1,n_up_samples))
        C[:,1] = interp1d(np.linspace(0,1,cont.shape[0]),
                          np.clip(cont[:,1],0,dims[1]))(
                              np.linspace(0,1,n_up_samples))
    C[:,0] = np.clip(C[:,0],0,dims[0])
    C[:,1] = np.clip(C[:,1],0,dims[1])
    H, xedges, yedges = np.histogram2d(C[:,0], C[:,1], bins=(np.sort(x), np.sort(y)))
    H = H>0
    return H.astype(bool)

def contour_to_mask(x,y,dims,extent = None,n_up_samples = 2000):
    '''
        H = contour_to_mask(x,y,dims,extent = None,n_up_samples = 2000)
    
        Create a mask from a contour
        
    '''

    H = contour_to_im(x=x, y=y, 
                      dims = dims,
                      extent = extent,
                      n_up_samples = n_up_samples)    
    # fix border cases
    #if np.sum(H[0,:]):
    #    H[0,:] = np.uint8(1)
    #if np.sum(H[-1,:]):
    #    H[-1,:] = np.uint8(1)
    #if np.sum(H[:,0]):
    #    H[:,0] = np.uint8(1)
    #if np.sum(H[:,-1]):
    #    H[:,-1] = np.uint8(1)
    from scipy.ndimage import morphology
    H = morphology.binary_dilation(H)
    H = morphology.binary_fill_holes(H)
    H = morphology.binary_erosion(H)
    H[0,:] = 0
    H[-1,:] = 0
    H[:,0] = 0
    H[:,-1] = 0
    return H.astype(bool)


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


def zipdir(path, outputpath):
    import zipfile
    with zipfile.ZipFile(outputpath, 'w', 
                         allowZip64 = True) as zipf:   
        for root, dirs, files in os.walk(path):
            for file in tqdm(files,desc = 'Compressing'):
                zipf.write(os.path.join(root, file),file)

def zipfiles(files, outputpath):
    import zipfile
    with zipfile.ZipFile(outputpath, 'w', 
                         allowZip64 = True) as zipf:   
        for f in tqdm(files,desc = 'Compressing'):
            zipf.write(f,os.path.basename(f))
