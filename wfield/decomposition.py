#  wfield - tools to analyse widefield data - decomposition 
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

from .utils import *
from .imutils import mask_to_3d
from .io import load_binary_block
from numpy.linalg import svd

def approximate_svd(dat, frames_average,
                    onsets = None,
                    k=200, 
                    mask = None,
                    nframes_per_bin = 15,
                    nbinned_frames = 5000,
                    nframes_per_chunk = 500,
                    divide_by_average = True):
    '''
    Approximate single value decomposition by estimating U from the average movie and using it to compute S.VT.
    This is similar to what described in Steinmetz et al. 2017

    Joao Couto - March 2020
    
    This computes the mean centered SVD of the dataset, it does not compute F-F0/F0 a.k.a. df/f.
    Compute it after using the SVD components.

    Inputs:
        dat (array)              : (NFRAMES, NCHANNEL, H, W) 
        k (int)                  : number of components to estimate (200)
        nframes_per_bin (int)    : number of frames to estimate the initial U components
        nbinned_frames (int)     : maximum number frames to estimate the initial U components
        nframes_per_chunk (int)  : window size to load to memory each time.
        divide_by_average (bool) : True   
    Returns:
        U   (array)             : 
        SVT (array)             : 
    '''
    from sklearn.preprocessing import normalize

    if hasattr(dat,'filename'):
        dat_path = dat.filename
    else:
        dat_path = None
    dims = dat.shape[1:]

    # the number of bins needs to be larger than k because of the number of components.
    if nbinned_frames < k:
        nframes_per_bin = np.clip(int(np.floor(len(dat)/(k))),1,nframes_per_bin)

    nbinned_frames = np.min([nbinned_frames,
                             int(np.floor(len(dat)/nframes_per_bin))])
    
    idx = np.arange(0,nbinned_frames*nframes_per_bin,nframes_per_bin,
                    dtype='int')
    if not idx[-1] == nbinned_frames*nframes_per_bin:
        idx = np.hstack([idx,nbinned_frames*nframes_per_bin-1])
    binned = np.zeros([len(idx)-1,*dat.shape[1:]],dtype = 'float32')
    for i in tqdm(range(len(idx)-1),desc='Binning raw data'):
        if dat_path is None:
            blk = dat[idx[i]:idx[i+1]] # work when data are loaded to memory
        else:
            blk = load_binary_block((dat_path,idx[i],nframes_per_bin),
                                    shape=dims)
        avg = get_trial_baseline(idx[i],frames_average,onsets)
        if divide_by_average:
            binned[i] = np.mean((blk-(avg + np.float32(1e-10)))
                                / (avg + np.float32(1e-10)), axis=0)
        else:
            binned[i] = np.mean(blk-(avg + np.float32(1e-10)), axis=0)
        if not mask is None:
            b = binned[i]
            mmask = mask_to_3d(mask,[dat.shape[1],*mask.shape])
            b[mmask == 0] = 0

    binned = binned.reshape((-1,np.multiply(*dims[-2:])))

    # Get U from the single value decomposition 
    cov = np.dot(binned,binned.T)/binned.shape[1]
    cov = cov.astype('float32')

    u,s,v = svd(cov)
    U = normalize(np.dot(u[:,:k].T, binned),norm='l2',axis=1)
    k = U.shape[0]     # in case the k was smaller (low var)
    # if trials are defined, then use them to chunk data so that the baseline is correct
    if onsets is None:
        idx = np.arange(0,len(dat),nframes_per_chunk,dtype='int')
    else:
        idx = onsets
    if not idx[-1] == len(dat):
        idx = np.hstack([idx,len(dat)-1])
    V = np.zeros((k,*dat.shape[:2]),dtype='float32')
    # Compute SVT
    for i in tqdm(range(len(idx)-1),desc='Computing SVT from the raw data'):
        if dat_path is None:
            blk = dat[idx[i]:idx[i+1]] # work when data are loaded to memory
        else:
            blk = load_binary_block((dat_path,idx[i],idx[i+1]-idx[i]),
                                shape=dims).astype('float32')
        avg = get_trial_baseline(idx[i],frames_average,onsets).astype('float32')
        blk = blk-(avg+np.float32(1e-10))
        if divide_by_average:
            blk /= avg+np.float32(1e-10)
        V[:,idx[i]:idx[i+1],:] = np.dot(
            U, blk.reshape([-1,np.multiply(*dims[1:])]).T).reshape((k,-1,dat.shape[1]))
    SVT = V.reshape((k,-1))
    U = U.T.reshape([*dims[-2:],-1])
    return U,SVT


def svd_blockwise(dat,frames_average,
                  k = 200, block_k = 20,
                  blocksize=120, overlap=8,
                  divide_by_average = True,
                  random_state=42):
    '''
    Computes the blockwise single value decomposition for a matrix that does not fit in memory.

    U,SVT,S,(block_U,block_SVT,blocks) = svd_blockwise(dat,
                                                   frames_average,
                                                   k = 200, 
                                                   block_k = 20,
                                                   blocksize=120, 
                                                   overlap=8)
    dat is a [nframes X nchannels X width X height] array
    frames_average is a [nchannels X width X height] array; the average to be subtracted before computing the SVD
    k is the number of components to be extracted (randomized SVD)

    The blockwise implementation works by first running the SVD on overlapping chunks of the movie. Secondly,  SVD is ran on the extracted temporal components and the spatial components are scaled to match the actual frame size. 
The chunks have all samples in time but only a fraction of pixels.

    This is adapted from matlab code by Simon Musall.
    A similar approach is described in Stringer et al. Science 2019.

    Joao Couto - March 2020
    '''
    from sklearn.utils.extmath import randomized_svd
    from sklearn.preprocessing import normalize

    nframes,nchannels,w,h = dat.shape
    n = nframes*nchannels
    # Create the chunks where the SVD is ran initially, 
    #these have all samples in time but only a few in space 
    #chunks contain pixels that are nearby in space 
    blocks = make_overlapping_blocks((w,h),blocksize=blocksize,overlap=overlap)
    nblocks = len(blocks)
    # M = U.S.VT
    # U are the spatial components in this case
    block_U = np.zeros((nblocks,blocksize,blocksize,block_k),dtype=np.float32)
    block_U[:] = np.nan
    # V are the temporal components
    block_SVT = np.zeros((nblocks,block_k,n),dtype=np.float32)
    block_U[:] = np.nan
    # randomized svd is ran on each chunk 
    for iblock,(i,j) in tqdm(enumerate(blocks), total= len(blocks),
                             desc= 'Computing SVD on data chunks:'):
        # subtract the average (this should be made the baseline instead)
        arr = np.array(dat[:,:,i[0]:i[1],j[0]:j[1]],dtype='float32')
        arr -= frames_average[:,i[0]:i[1],j[0]:j[1]]
        if divide_by_average:
            arr /= frames_average[:,i[0]:i[1],j[0]:j[1]]
        bw,bh = arr.shape[-2:]
        arr = arr.reshape([-1,np.multiply(*arr.shape[-2:])])
        u, s, vt = randomized_svd(arr.T,
                                  n_components=block_k,
                                  n_iter=5,
                                  power_iteration_normalizer ='LQ',
                                  random_state=random_state)
        block_U[iblock,:bw,:bh,:] = u.reshape([bw,bh,-1])
        block_SVT[iblock] = np.dot(np.diag(s),vt)

    U,SVT,S = _complete_svd_from_blocks(block_U,block_SVT,blocks,k,(w,h))    
    return U,SVT,S,(block_U,block_SVT,blocks)      


def _complete_svd_from_blocks(block_U,block_SVT,blocks,k,dims,
                              n_iter=15,random_state=42):
    # Compute the svd of the temporal components from all blocks
    from sklearn.utils.extmath import randomized_svd
    u, s, vt = randomized_svd(
        block_SVT.reshape([np.multiply(*block_SVT.shape[:2]),-1]),
        n_components=k,
        n_iter=n_iter,
        power_iteration_normalizer ='QR',
        random_state=random_state)
    S = s
    SVT = np.dot(np.diag(S),vt)
    # Map the blockwise spatial components components to the second SVD 
    U = np.dot(assemble_blockwise_spatial(block_U,blocks,dims),u)
    return U,SVT,S


def assemble_blockwise_spatial(block_U,blocks,dims):
    w,h = dims
    U = np.zeros([block_U.shape[0],block_U.shape[-1],w,h],dtype = 'float32')
    weights = np.zeros((w,h),dtype='float32')
    for iblock,(i,j) in enumerate(blocks):
        lw,lh = (i[1]-i[0],j[1]-j[0])
        U[iblock,:,i[0]:i[1],j[0]:j[1]] = block_U[iblock,:lw,:lh,:].transpose(-1,0,1)
        weights[i[0]:i[1],j[0]:j[1]] += 1
    U = (U/weights).reshape((np.multiply(*U.shape[:2]),-1))
    return U.T


#############################
############ PMD ############
#############################

# This was written by Ian Kinsella for the NeuroCAAS implementation
# of the penalized matrix factorization of widefield data with 2 channels.
# The interface was simplified to be used locally
# It may require alot of memory

def eval_spatial_stats(u):
    num_pixels, num_comps = np.prod(u.shape[:-1]), u.shape[-1]
    vert_diffs = np.abs(u[1:,:,:] - u[:-1,:,:])
    horz_diffs = np.abs(u[:,1:,:] - u[:,:-1,:])
    avg_diff = ((np.sum(vert_diffs.reshape((-1, num_comps), order='F'), axis=0) + 
                 np.sum(horz_diffs.reshape((-1, num_comps), order='F'), axis=0)) /
                (np.prod(vert_diffs.shape[:-1]) + np.prod(horz_diffs.shape[:-1])))
    avg_elem = np.mean(np.abs(u.reshape((num_pixels,-1), order='F')), axis=0)
    return avg_diff / avg_elem


def eval_temporal_stats(v):
    return np.mean(np.abs(v[:, :-2] + v[:, 2:] - 2*v[:, 1:-1]), axis=-1) / np.mean(np.abs(v), axis=-1)

def noise_decomp(num_components, params=None):
    """ TESTING: downsampled case """
    E = np.random.randn(params['block_height'], params['block_width'], params['window_length'])
    U = np.zeros((params['block_height'] * params['block_width'] * num_components,), dtype=np.float64)
    V = np.zeros((params['window_length'] * num_components,), dtype=np.float64)

    if params['d_sub'] > 1 or params['t_sub'] > 1:
        # Downsample Simulated Noise
        E_sub = skimage.measure.block_reduce(E, (d_sub, d_sub, t_sub), func=np.mean)

        # Perform Partial Decomposition Without Thresholds W/ Decimated Init
        _ = decimated_decompose(params['block_height'],
                                params['block_width'],
                                params['d_sub'],
                                params['window_length'], 
                                params['t_sub'],
                                np.reshape(E, (-1,)), np.reshape(E_sub, (-1,)),
                                U, V,
                                1e3, 1e3,
                                num_components, num_components,
                                params['max_iters_main'], params['max_iters_init'],
                                params['tol'])
    else:
        # Perform Partial Decomposition Without Thresholds
        _ = decompose(params['block_height'],
                      params['block_width'],
                      params['window_length'], 
                      np.reshape(E, (-1,)),
                      U, V,
                      1e3, 1e3,
                      num_components, num_components,
                      params['max_iters_main'], params['max_iters_init'],
                      params['tol'])

    # Return Summary Stats
    return (eval_spatial_stats(U.reshape((params['block_height'], params['block_width'], -1), order='F')),
            eval_temporal_stats(V.reshape((-1, params['window_length']))))
