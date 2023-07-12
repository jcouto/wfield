#  wfield - tools to analyse widefield data - tools to match between different sessions
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
from .registration import registration_ecc

def get_transform(template, frame):
    '''
    Gets the transform between the frames and the template.
    M,transform, transform_inv = get_transform(template, frame)
    
        - M is the affine transform matrix (opencv format)
        - transform is a transform funtion (input a frame and it returns the transformed frame)
        - tranform_inv is the inverted
        
    Joao Couto - wfield (2023)
    '''
    M,res = registration_ecc(frame.astype('float32'),template.astype('float32'),
                                          warp_mode = cv2.MOTION_AFFINE)
    h,w = res.shape
    transform = partial(cv2.warpAffine, M = M, dsize = (w,h),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    transform_inv = partial(cv2.warpAffine, M = M, dsize = (w,h),
                             flags=cv2.INTER_LINEAR)
    
    
    return  M,transform,transform_inv


def match_sessions(template_avg, avg, lmarks = None, template_mask = None):
    '''
    Match 2 sessions.
    This will return the new landmarks and mask for a session.
    Inputs:
        - template_avg - single channel frame with the template session to use
        - temlate_mask - decomposition mask used in the template 
        - lmarks - landmarks dictionary 
        - avg - single channel frame to match
        
    Outputs:
        -  M      - affine transform between sessions
        - nlmarks - landmarks dictionary
        - nmask   - transformed decomposition mask 
        
    Joao Couto - wfield (2023)
    
    '''
    # get the transforms
    M,trans,trans_inv = get_transform(template_avg, avg)
    # Map these to the AffineTransform object
    from skimage.transform import AffineTransform
    iM  = AffineTransform(matrix = np.vstack([cv2.invertAffineTransform(M),[0,0,1]]))
    M  = AffineTransform(matrix = np.vstack([M,[0,0,1]]))
    nlmarks = None
    nmask = None
    if not lmarks is None:
        # create the new landmarks
        nlmarks = lmarks.copy()
        nlmarks['transform'] = M
        nlmarks['transform_inverse'] = iM
        # transform the points
        nlmarks['transform_type'] = 'affine'
        nlmarks['landmarks_match'] = lmarks['landmarks_match'].copy()
        
        (nlmarks['landmarks_match'].x,nlmarks['landmarks_match'].y) = apply_affine_to_points(
            lmarks['landmarks_im'].x, lmarks['landmarks_im'].y, nlmarks['transform'])
    if not template_mask is None:
        # create the new mask
        nmask = im_apply_transform(template_mask,iM)  
    return M, nlmarks, nmask


def prepare_multisession_match_files(session_folder, template_folder):
    '''
    This prepares the files to match between sessions. 
    
    Run this after motion correction on the session you want to match.
    
    transform,nlmarks,nmask = prepare_multisession_match_files(session_folder, template_folder)
    
    Joao Couto - wfield (2023)

    '''
    
    template_average_path = glob(pjoin(template_folder, 'frames_average.npy'))

    template = None
    mask = None
    lmarks = None
    from .allen import load_allen_landmarks,save_allen_landmarks
    if len(template_average_path):
        template_nt = np.load(template_average_path[0])[0].squeeze()
        # transform if possible:
        lmarkfile = glob(pjoin(template_folder, '*landmarks.json'))
        if len(lmarkfile):
            lmarks = load_allen_landmarks(lmarkfile[0])
            template = im_apply_transform(template_nt, lmarks['transform']).astype('uint16')
        maskfile = glob(pjoin(template_folder, 'mask.npy'))
        if len(maskfile):
            mask = np.load(maskfile[0])
            if not lmarks is None:
                mask = (im_apply_transform(mask, lmarks['transform'])>0).astype('uint8')
                
    # try to load the mask
    frames_path = glob(pjoin(session_folder, 'frames_average.npy'))
    if len(frames_path):
        frames = np.load(frames_path[0])[0].squeeze()

    transform, nlmarks, nmask = match_sessions(template_avg=template, avg=frames,
                                               lmarks = lmarks, template_mask = mask)
    
    save_allen_landmarks(filename = pjoin(session_folder, os.path.basename(lmarkfile[0])), **nlmarks)
    
    np.save(pjoin(session_folder,'mask.npy'), nmask.astype(bool))
    
    return transform, nlmarks, nmask
