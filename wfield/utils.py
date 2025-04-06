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
import warnings
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

from joblib import Parallel, delayed
    
from functools import partial
from scipy.interpolate import interp1d
from scipy.sparse import load_npz, issparse,csr_matrix
from scipy.ndimage import binary_fill_holes,binary_erosion
from skimage.filters import gaussian 
from skimage.morphology import remove_small_objects

print = partial(print, flush=True)


# create the wfield folder and place the references.
# this runs if installed from pip
wfield_dir = pjoin(os.path.expanduser('~'),'.wfield')

def _create_wfield_folder():
    if not os.path.isdir(wfield_dir):
        print('Created {0}'.format(wfield_dir))
        os.makedirs(wfield_dir)
    # try first from the shared folder 
    modulepath = pjoin(__file__.split('lib')[0],'share','wfield')
    refpath = pjoin(modulepath, 'references')
    if os.path.exists(refpath):
        reference_files = [pjoin(refpath,r) for r in os.listdir(refpath)]
        from shutil import copyfile
        for f in reference_files:
            if os.path.isfile(f):
                copyfile(f,f.replace(refpath,wfield_dir))
    else:
        try:
            import requests
        except Exception as err:
            print(err)
            raise(OSError('Could not import the requests package, please install it "pip install requests"'))
        from shutil import copyfileobj
        webpath = 'https://raw.githubusercontent.com/jcouto/wfield/master/references/{0}'
        files = ['dorsal_cortex_ccf_labels.json',
                 'dorsal_cortex_landmarks.json',
                 'dorsal_cortex_outline.npy',
                 'dorsal_cortex_projection.npy',
                 'vis_ccf_labels.json',
                 'vis_outline.npy',
                 'vis_projection.npy']
        # download the files
        for f in files:
            print('    Downloading {0}'.format(f))
            if '.json' in f: # because of the encodings.
                with open(pjoin(wfield_dir,f),'w') as fid:
                    res = requests.get(webpath.format(f))
                    fid.write(res.text)
                    del res
            else:
                with open(pjoin(wfield_dir,f),'wb') as fid:
                    raw = requests.get(webpath.format(f),stream=True)
                    copyfileobj(raw.raw, fid)
                    del raw
                
def estimate_similarity_transform(ref,points):
    '''
    
    ref = np.vstack([landmarks_im['x'],landmarks_im['y']]).T
    match = point_stream.data    
    cor = np.vstack([match['x'],match['y']]).T
    
    M = estimate_similarity_transform(ref, cor)
    
    Joao Couto - wfield (2020)
    '''
    from skimage.transform import SimilarityTransform
    M = SimilarityTransform()
    M.estimate(ref,points)
    return M

def apply_affine_to_points(x,y,M):
    '''
    Apply an affine transform to a set of contours or (x,y) points.

    x,y = apply_affine_to_points(x, y, tranform)

    Joao Couto - wfield (2020)
    '''
    if M is None:
        nM = np.identity(3,dtype = np.float32)
    else:
        nM = M.params
    xy = np.vstack([x,y,np.ones_like(y)])
    res = (nM @ xy).T
    return res[:,0],res[:,1]


def im_adapt_hist(im,clip_limit = .1, grid_size=(8,8)):
    ''' 
    Adaptative histogram of image

        eqim = im_adapt_hist(im,.1)

    Joao Couto - wfield, 2020
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(im.squeeze())

def im_apply_transform(im,M,dims = None):
    '''
    Applies an affine transform M to an image.
    nim = im_apply_transform(im,M)

    Joao Couto - wfield, 2020
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
                   clip = True,
                   preserve_range = True)
        return csr_matrix(tmp.reshape(shape))
    else:    
        return warp(im,M,
                    order = 1,
                    mode='constant',
                    cval = 0,
                    clip=True,
                    preserve_range=True)

def lowpass(X, w = 7.5, fs = 30.):
    '''
    Apply a lowpass filter to data.
        - w is the filter frequency
        - fs is the sampling rate of the signal.

    Usage:
    
    Xfiltered = lowpass(X, w = 7.5, fs = 30.)

    Joao Couto - 2020 
    '''
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen = 50)

def highpass(X, w = 3., fs = 30.):
    '''
    Apply a highpass filter to data.
        - w is the filter frequency
        - fs is the sampling rate of the signal.

    Usage:
    
    Xfiltered = highpass(X, w = 3., fs = 30.)

    Joao Couto - wfield, 2020
    '''
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen = 50)

def analog_ttl_to_onsets(dat,time=None, mfilt=3):
    '''
    Extract onsets from an analog TTL signal.

    Use this to get the onsets and offsets of a digital signal.
      - dat a 1d signal with the TTL
      - time (None) will return in samples if None or sec if a vector of frame times
      - mfilt (3) 
 
    Usage:
        onsets,offsets = analog_ttl_to_onsets(dat)

     Joao Couto - wfield, 2020
    '''
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

    Joao Couto - wfield, 2020
    '''
    chunks = np.arange(0,nframes,chunksize,dtype = int)
    if (nframes - chunks[-1]) < min_chunk_size:
        chunks[-1] = nframes
    if not chunks[-1] == nframes:
        chunks = np.hstack([chunks,nframes])
    return [[chunks[i],chunks[i+1]] for i in range(len(chunks)-1)]


def make_overlapping_blocks(dims, blocksize = 128, overlap = 16):
    '''
    Creates overlapping block indices to span an image
    Use this to split an image and iterate over different blocks.

    Usage:
        blocks = make_overlapping_blocks(dims, blocksize = 128, overlap = 16)

    Joao Couto - wfield, 2020
    '''
    
    w,h=dims
    blocks = []
    for i,a in enumerate(range(0,w,blocksize-overlap)):
        for j,b in enumerate(range(0,h,blocksize-overlap)):
            blocks.append([(a,np.clip(a+blocksize,0,w)),(b,np.clip(b+blocksize,0,h))])
    return blocks

def reconstruct(u,svt,dims = None):
    '''
    Reconstruct a decomposed signal (e.g. decomposed with SVD for example).

    Can also reconstruct sparse arrays, use dims when reconstructing sparse.

    Usage:
        res = reconstruct(u,svt,dims = None)
    
    Joao Couto - wfield, 2020
    '''
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]  
    return (u@svt).reshape((*dims,-1)).transpose(-1,0,1).squeeze()

def _apply_function_single_pix(U,SVT,func):
    '''
    Apply a function to a single pixel (helper to run in parallel)

    Joao Couto - wfield, 2021
    '''
    return func(U@SVT)

def apply_pixelwise_svd(U,SVT,func, nchunks = 1024):
    '''
    Map a function to every pixel by reconstructing the SVD 

    Usage:
        variance_map = apply_pixelwise_svd(U, SVT, partial(np.nanvar,axis=1), nchunks = 1024)
        
        mean_map = apply_pixelwise_svd(U, SVT, partial(np.nanmean,axis=1), nchunks = 1024)

    Joao Couto - wfield, 2021
    '''
    dims = U.shape

    U = U.reshape([-1,dims[-1]])
    npix = U.shape[0]

    idx = np.array_split(np.arange(0,npix),nchunks)

    res = runpar(_apply_function_single_pix,
                 [U[ind,:] for ind in idx],
                 SVT=SVT,
                 func = func)
    res = np.hstack(res).astype('float32')
    U = U.reshape(dims)
    return res.reshape(dims[:2])


class SVDStack(object):
    def __init__(self, U, SVT, dims = None,
                 warped = None,
                 M = None, dtype = 'float32',nchunks = 1054):
        '''
stack = SVDStack(U,SVT)

Treat a decomposed dataset like a numpy array.
Args:
        - U: spatial components
        - SVT: temporal components
        - dims: dimensions of the dataset [H,W] (for loading sparse matrices)
        - warped: warped spatial components (e.g. aligned to a reference atlas)
        - M: transform to warp spatial components
        - dtype: cast to this datatype
        - nchunks: number of chunks for pixelwise analysis

        Joao Couto - wfield, 2020
        '''
        self.U = U.astype(dtype)
        self.SVT = SVT.astype(dtype)
        self.nchunks = nchunks
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
        self.M = M
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
                                                        M = self.M)).transpose([1,2,0]).astype(np.float32)
            if not self.U_warped is None:
                self.U = self.U_warped
                self.warped = True
        self.Uflat = self.U.reshape(-1,self.U.shape[-1])

    def mean(self):
        '''Pixelwise mean of the stack '''
        return apply_pixelwise_svd(self.U, self.SVT, partial(np.nanmean,axis=1), nchunks = self.nchunks)

    def var(self):
        '''Pixelwise variance of the stack '''
        return apply_pixelwise_svd(self.U, self.SVT, partial(np.nanvar,axis=1), nchunks = self.nchunks)

    def std(self):
        '''Pixelwise standard deviation of the stack '''
        return apply_pixelwise_svd(self.U, self.SVT, partial(np.nanstd,axis=1), nchunks = self.nchunks)
    
    def __len__(self):
        return self.SVT.shape[1]
    
    def __getitem__(self,*args):
        ndims  = len(args)
        if type(args[0]) is slice:
            idxz = range(*args[0].indices(self.shape[0]))
        else:
            idxz = args[0]      
        return reconstruct(self.Uflat,self.SVT[:,idxz],dims = self.shape[1:])
    
    def get_timecourse(self,xy):
        ''' Get a timecourse for the specified indices. 
Index are in xy, like what np.where(mask) returns
timecourse = get_timecourse([x,y])

or 

timecourse = np.nanmean(get_timecourse([x,y]),axis = 1)
'''
        x = np.array(np.clip(xy[0],0,self.shape[1]),dtype=int)
        y = np.array(np.clip(xy[1],0,self.shape[2]),dtype=int)
        idx = np.ravel_multi_index((x,y),self.shape[1:])
        t = self.Uflat[idx,:]@self.SVT
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
    
    Usage:
    
    point = [6.5,3.0]
    region,side,index = point_find_ccf_region(point,refregions)
    
    Joao Couto - wfield, 2020
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
    Imprint a contour on an image.
        see also: contour_to_mask

    Usage:
     
        im = contour_to_im(x,y,dims,extent=None,n_up_samples = 1000)
    
    Joao Couto - wfield, 2020
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
    Create a mask from a contour
    
    Usage:    
        H = contour_to_mask(x,y,dims,extent = None,n_up_samples = 2000)
    
    Joao Couto - wfield, 2020        
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


def runpar(f,X,nprocesses = None,desc = None,**kwargs):
    ''' 
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine (default caps at 8)
                 **kwargs)          # additional arguments passed to the function (dictionary)

    Joao Couto - wfield, 2020

    '''
    if nprocesses is None:
        nprocesses = 8
    if desc is None:
        res = Parallel(n_jobs = nprocesses)(delayed(f)(x,**kwargs) for x in X)
    else:
        from tqdm import tqdm
        res = Parallel(n_jobs = nprocesses)(delayed(f)(x,**kwargs) for x in tqdm(X,desc = desc))
    return res


def get_std_mask(ch1data, nframes = 1000, filter_sigma = 10,threshold = 60,minsize = 5000):
    '''
    Returns a mask based on the standard deviation of a fraction of a movie.

    - ch1data: N,W,H array
    - nframes: default 1000, number of frames to downsample
    - filter_sigma: default 10, size of the gaussian filter applied after std
    - threshold : default 60, threshold in percentile units
    - minsize: default 5000, minimum size of objects used in the mask

Usage:
    tt = get_std_mask(dat[:,0],threshold=60)

Joao Couto - wfield 2023
    '''
    selidx = np.sort(np.random.choice(np.arange(ch1data.shape[0]),size = nframes, replace=False))
    stdproj = np.std(ch1data[selidx],axis=0).squeeze()

    tt = gaussian(stdproj, sigma=filter_sigma)
    tt = tt>np.percentile(tt,threshold)
    tt = binary_erosion(tt)
    tt = remove_small_objects(tt,minsize)
    tt = binary_fill_holes(tt)
    mask = tt.copy().astype(np.float32)
    # mask = gaussian(mask,sigma = 30)
    # mask[tt==1] = 1
    return mask

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
