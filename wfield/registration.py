#  wfield - tools to analyse widefield data - motion correction 
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

from  .utils import *
from skimage.transform import AffineTransform
cv2.setNumThreads(1)

def findTransformECC(template,dst,M,warp_mode,criteria,inputMask,gaussFiltSize):
    return cv2.findTransformECC(template,dst,
                                M, warp_mode,
                                criteria,
                                inputMask=inputMask,
                                gaussFiltSize=gaussFiltSize)

cv2ver = cv2.__version__.split('.')
if (int(cv2ver[0]) == 3) and (int(cv2ver[1]) <= 4):
    if int(cv2ver[2]) <= 5:
        def findTransformECC(template,
                             dst,
                             M,
                             warp_mode,
                             criteria,
                             inputMask,
                             gaussFiltSize):
            return cv2.findTransformECC(template,dst,
                                        M, warp_mode,
                                        criteria,
                                        inputMask=inputMask)
elif (int(cv2ver[0]) == 4) and (int(cv2ver[1]) <= 1):
    # gaussFiltSize is a mandatory input on opencv 4.4 but not 4.1
    def findTransformECC(template,
                         dst,
                         M,
                         warp_mode,
                         criteria,
                         inputMask,
                         gaussFiltSize):
        return cv2.findTransformECC(template,dst,
                                    M, warp_mode,
                                    criteria,
                                    inputMask=inputMask)
def registration_ecc(frame,template,
                     niter = 25,
                     eps0 = 1e-2,
                     warp_mode = cv2.MOTION_EUCLIDEAN,
                     prepare = True,
                     gaussian_filter = 1,
                     hann = None,
                     **kwargs):
    h,w = template.shape
    if hann is None:
        hann = cv2.createHanningWindow((w,h),cv2.CV_32FC1)
        hann = (hann*255).astype('uint8')
    dst = frame.astype('float32')
    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                niter,  eps0)
    (res, M) = findTransformECC(template,dst,
                                M, warp_mode,
                                criteria,
                                inputMask=hann, gaussFiltSize=gaussian_filter)
    dst = cv2.warpAffine(frame, M, (w,h),
                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    return M, np.clip(dst,0,(2**16-1)).astype('uint16')


def _xy_rot_from_affine(affines):
    '''
    helper function to parse affine parameters from ECC
    '''
    xy = []
    rot = []
    for r in affines:
        M = np.vstack([r, np.array([0,0,1])])
        M = AffineTransform(M)
        xy.append(M.translation)
        rot.append(M.rotation)
    rot = np.rad2deg(np.array(rot))
    xy = np.array(xy)
    return xy,rot

def registration_upsample(frame,template):
    h,w = frame.shape
    dst = frame.astype('float32')
    (xs, ys), sf = cv2.phaseCorrelate(template.astype('float32'),dst)    
    M = np.float32([[1,0,xs],[0,1,ys]])
    dst = cv2.warpAffine(dst,M,(w, h))
    return (xs,ys),(np.clip(dst,0,(2**16-1))).astype('uint16')

def _register_multichannel_stack(frames,templates,mode='2d',
                                 niter = 25,
                                 eps0 = 1e-3,
                                 warp_mode = cv2.MOTION_EUCLIDEAN): # mode 2d

    nframes, nchannels, h, w = frames.shape     
    if mode == 'ecc':
        hann = cv2.createHanningWindow((w,h),cv2.CV_32FC1)
        hann = (hann*255).astype('uint8')

    ys = np.zeros((nframes,nchannels),dtype=np.float32)
    xs = np.zeros((nframes,nchannels),dtype=np.float32)
    rot = np.zeros((nframes,nchannels),dtype=np.float32)
    stack = np.zeros_like(frames,dtype = 'uint16')
    for ichan in range(nchannels):
        chunk = frames[:,ichan].squeeze()
        if mode == '2d':
            res = runpar(registration_upsample, chunk,
                         template = templates[ichan])
            ys[:,ichan] = np.array([r[0][1] for r in res],dtype='float32')
            xs[:,ichan] = np.array([r[0][0] for r in res],dtype='float32')

        elif mode == 'ecc':
            res = runpar(registration_ecc, chunk,
                         template = templates[ichan],
                         hann = hann,
                         niter = niter,
                         eps0 = eps0,
                         warp_mode = warp_mode)
            xy,rots = _xy_rot_from_affine([r[0] for r in res])
            ys[:,ichan] = xy[:,1]
            xs[:,ichan] = xy[:,0]
            rot[:,ichan] = rots
        stack[:,ichan,:,:] = np.stack([r[1] for r in res])
    return (xs,ys,rot), stack

def motion_correct(dat, out = None, chunksize=512, nreference = 60, mode = 'ecc', apply_shifts=True):
    '''
    Motion correction by translation.
    This estimate x and y shifts using phase correlation. 
    
    The reference image is the average of the chunk in the center.

    Inputs:
        dat (array)           : (NFRAMES, NCHANNEL, H, W) is overwritten if apply_shifts is True
        chunksize (int)       : size of the chunks (needs to be small enough to fit in memory - default 512)
        nreference            : number of frames to take as reference (default 60)
        apply_shifts          : overwrite the data with the motion corrected (default True)
    Returns:
        yshifts               : shitfs in y (NFRAMES, NCHANNELS)
        xshifts               : shifts in x
    '''
    nframes,nchan,h,w = dat.shape
    if out is None:
        out = dat
    chunks = chunk_indices(nframes,chunksize)
    xshifts = []
    yshifts = []
    rshifts = []
    # reference is from the start of the file (nreference frames to nreference*2)
    # (chunksize frames and for each channel independently)
    nreference = int(nreference)
    chunk = np.array(dat[nreference:nreference*2])
    refs = chunk[0].astype('float32')
    # align to the ref of each channel and use the mean
    _,refs = _register_multichannel_stack(chunk,refs,mode=mode)
    refs = np.mean(refs,axis=0).astype('float32')
    for c in tqdm(chunks,desc='Motion correction'):
        # this is the reg bit
        localchunk = np.array(dat[c[0]:c[-1]])
        # always do 2d first
        #(xs,ys,rot),corrected = _register_multichannel_stack(localchunk,refs,
        #                                                     mode='2d')
        #if not mode == '2d' :
        #xs += xs0
        #ys += ys0
        (xs,ys,rot),corrected = _register_multichannel_stack(
            localchunk,
            refs,
            mode=mode)
        if apply_shifts:
            out[c[0]:c[-1]] = corrected[:]
        yshifts.append(ys)
        xshifts.append(xs)
        rshifts.append(rot)
    return (np.vstack(yshifts),np.vstack(xshifts)),np.vstack(rshifts)


