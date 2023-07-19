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

def compute_locaNMF(U,V,atlas,brain_mask,
                    minrank = 1, # rank = how many components per brain region.
                    maxrank = 10, #Set maxrank to around 10 for regular dataset.
                    min_pixels = 100, # minimum number of pixels in Allen map for it to be considered a brain region
                    loc_thresh = 70, # Localization threshold, i.e. percentage of area restricted to be inside the 'atlas boundary'
                    r2_thresh = 0.99, # Fraction of variance in the data to capture with LocaNMF
                    nonnegative_temporal = False, # Do you want nonnegative temporal components? The data itself should also be nonnegative in this case.
                    maxiter_lambda = 300,
                    device = 'auto',
                    verbose = [True, False, False]):
    '''
This function runs locaNMF from wfield analysis outputs.
It uses the original package for LocaNMF, written by Ian Kinsella and Shreya Saxena
Reference: 
    Saxena S, Kinsella I, Musall S, Kim SH, Meszaros J, et al. (2020) 
    Localized semi-nonnegative matrix factorization (LocaNMF) of widefield calcium imaging data. 
    PLOS Computational Biology 16(4): e1007791. https://doi.org/10.1371/journal.pcbi.1007791

Usage:
    
    A,C,regions = compute_locaNMF(U,V,atlas,brain_mask,
                    minrank = 1, 
                    maxrank = 10, 
                    min_pixels = 100,
                    loc_thresh = 70, 
                    r2_thresh = 0.99,
                    device = 'cuda')
    
    Joao Couto - wfield, 2023
    '''
    try:
        from locanmf import LocaNMF
    except Exception as err:
        print(err)
        raise(OSError("This analysis requires the locaNMF package."))
    
    import torch
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print('torch could not find a cuda capable GPU, using the CPU (slower).')
            device = 'cpu'
            
    rank_range = (minrank, maxrank, 1)
    if nonnegative_temporal:
        r = V.T
    else:
        q, r = np.linalg.qr(V.T)
    video_mats = (np.copy(U[brain_mask]), r.T)
    del U

    region_mats = LocaNMF.extract_region_metadata(brain_mask, atlas, min_size=min_pixels)
    region_metadata = LocaNMF.RegionMetadata(region_mats[0].shape[0],
                                               region_mats[0].shape[1:],
                                               device=device)

    region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                        torch.from_numpy(region_mats[1]),
                        torch.from_numpy(region_mats[2].astype(np.int64)))

    # Do SVD
    if device=='cuda': torch.cuda.synchronize()
    region_videos = LocaNMF.factor_region_videos(video_mats,
                                                   region_mats[0],
                                                   rank_range[1],
                                                   device=device)
    if device=='cuda': torch.cuda.synchronize()
    low_rank_video = LocaNMF.LowRankVideo(
        (int(np.sum(brain_mask)),) + video_mats[1].shape, device=device)
    low_rank_video.set(torch.from_numpy(video_mats[0].T),
                       torch.from_numpy(video_mats[1]))
    if device=='cuda': torch.cuda.synchronize()
    locanmf_comps = LocaNMF.rank_linesearch(low_rank_video,
                                            region_metadata,
                                            region_videos,
                                            maxiter_rank = maxrank-minrank+1,
                                            maxiter_lambda = maxiter_lambda, 
                                            maxiter_hals = 20,
                                            lambda_step = 1.35,
                                            lambda_init = 1e-6, 
                                            loc_thresh = loc_thresh,
                                            r2_thresh = r2_thresh,
                                            rank_range = rank_range,
                                            nnt = nonnegative_temporal,
                                            verbose = verbose,
                                            sample_prop = (1,1),
                                            device = device)
    if device=='cuda': torch.cuda.synchronize()
    # Get LocaNMF spatial and temporal components
    A = locanmf_comps.spatial.data.cpu().numpy().T
    A_reshape = np.zeros((brain_mask.shape[0],brain_mask.shape[1],A.shape[1])); A_reshape.fill(np.nan)
    A_reshape[brain_mask,:] = A

    if nonnegative_temporal:
        C = locanmf_comps.temporal.data.cpu().numpy()
    else:
        C = np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T

    regions = region_metadata.labels.data[locanmf_comps.regions.data].cpu().numpy()

    if device=='cuda':
        torch.cuda.empty_cache()
    
    return A_reshape,C,regions

