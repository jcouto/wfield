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
                     niter = 5000,
                     eps0 = 1e-10,
                     warp_mode = cv2.MOTION_EUCLIDEAN,
                     prepare = True,
                     gaussian_filter = 1,
                     conv_kernel = None,
                     hann = None,
                     **kwargs):
    h,w = template.shape
    if hann is None:
        hann = cv2.createHanningWindow((w,h),cv2.CV_32FC1)
        hann = (hann*255).astype('uint8')
    dst = frame.astype('float32')
    if not conv_kernel is None:
        dst = _conv_frame(dst,conv_kernel)

    if warp_mode in [cv2.MOTION_HOMOGRAPHY]: # added support for perspective transforms
        M = np.eye(3, 3, dtype=np.float32)
    else:
        M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                niter,  eps0)
    (res, M) = findTransformECC(template,dst,
                                M, warp_mode,
                                criteria,
                                inputMask=hann, gaussFiltSize=gaussian_filter)
    if M.shape[0] == 3:
        dst = cv2.warpPerspective(frame.astype('float32'), M, (w,h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    else:
        dst = cv2.warpAffine(frame.astype('float32'), M, (w,h),
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

def registration_upsample(frame,template,conv_kernel = None):

    h,w = frame.shape
    dst = frame.astype('float32')
    if not conv_kernel is None:
        dst = _conv_frame(dst,conv_kernel)
    (xs, ys), sf = cv2.phaseCorrelate(template.astype('float32'),dst)    
    M = np.float32([[1,0,xs],[0,1,ys]])
    dst = cv2.warpAffine(frame.astype('float32'),M,(w, h))
    return (xs,ys),(np.clip(dst,0,(2**16-1))).astype('uint16')

def _register_multichannel_stack(frames,templates,mode='2d',
                                 niter = 100,
                                 eps0 = 1e-3,
                                 warp_mode = cv2.MOTION_EUCLIDEAN,
                                 conv_kernel = None): # mode 2d

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
                         template = templates[ichan],conv_kernel= conv_kernel)
            ys[:,ichan] = np.array([r[0][1] for r in res],dtype='float32')
            xs[:,ichan] = np.array([r[0][0] for r in res],dtype='float32')

        elif mode == 'ecc':
            res = runpar(registration_ecc, chunk,
                         template = templates[ichan],
                         hann = hann,
                         niter = niter,
                         eps0 = eps0,
                         warp_mode = warp_mode,
                         conv_kernel = conv_kernel)
            xy,rots = _xy_rot_from_affine([r[0] for r in res])
            ys[:,ichan] = xy[:,1]
            xs[:,ichan] = xy[:,0]
            rot[:,ichan] = rots
        stack[:,ichan,:,:] = np.stack([r[1] for r in res])
    return (xs,ys,rot), stack


def motion_correct(dat, out = None,
                   refs = None,
                   chunksize=512,
                   nreference = 60,
                   mode = '2d',
                   diff_gaussians_filter = [3,5],
                   apply_shifts=True):
    '''
    Motion correction by translation.
    This estimate x and y shifts using phase correlation. 
    
    The reference image is the average of the chunk in the center.

    Inputs:
        dat (array)           : (NFRAMES, NCHANNEL, H, W) is overwritten if apply_shifts is True
        out (array)           : same size as dat or None to overwrite dat
        refs (array)          : reference frames (NCHANNEL, H, W) or None to compute from nreference frames
        chunksize (int)       : size of the chunks (needs to be small enough to fit in memory - default 512)
        nreference            : number of frames to take as reference (default 60)
        apply_shifts          : overwrite the data with the motion corrected (default True)
        mode                  : ecc (default) is rigid body; 2d is only translation in x and y using dft
        diff_gaussians_filter : convolve with a difference of gaussians to highlight the blood vessels 
    Returns:
        (yshifts, xshifts)    : shitfs in y and x ((NFRAMES, NCHANNELS),(NFRAMES, NCHANNELS))
        rot_shifts            : rotational shifts if in ecc mode 
    '''
    nframes,nchan,h,w = dat.shape
    conv_kernel = None
    if not diff_gaussians_filter is None:
        conv_kernel = dog_kernel(*diff_gaussians_filter)
    if out is None:
        out = dat
    chunks = chunk_indices(nframes,chunksize)
    xshifts = []
    yshifts = []
    rshifts = []
    # reference is from the start of the file (nreference frames to nreference*2)
    # (chunksize frames and for each channel independently)
    if refs is None:
        nreference = int(nreference)
        chunk = np.array(dat[nreference:nreference*2])
        refs = chunk[0].astype('float32')
        if not conv_kernel is None:
            for iref,r in enumerate(refs):
                refs[iref] = _conv_frame(r,conv_kernel)
        # align to the ref of each channel and use the mean
        _,refs = _register_multichannel_stack(chunk,refs,mode=mode,conv_kernel = conv_kernel)
        refs = np.mean(refs,axis=0).astype('float32')
        if not conv_kernel is None: # then filter the template so it is not done every time
            for iref,r in enumerate(refs):
                refs[iref] = _conv_frame(refs[0],conv_kernel) # pass the same template to both channels if DoG convolved
    for c in tqdm(chunks,desc='Motion correction'):
        # this is the reg bit
        localchunk = np.array(dat[c[0]:c[-1]])
        (xs,ys,rot),corrected = _register_multichannel_stack(
            localchunk,
            refs,
            mode=mode,
            conv_kernel = conv_kernel)
        if apply_shifts:
            out[c[0]:c[-1]] = corrected[:]
            if hasattr(out,'flush'):
                out.flush() # write to disk
        yshifts.append(ys)
        xshifts.append(xs)
        rshifts.append(rot)
    return (np.vstack(yshifts),np.vstack(xshifts)),np.vstack(rshifts)


def _conv_frame(frame,kernel = None):
    '''
    Convolve frame with a kernel.
    does difference of gaussians if the type of kernel is list and len is 2
    returns float32
    '''
    if kernel is None:
        return frame
    if len(kernel) == 2 and type(kernel) is list:
        # then lets create the kernel
        kernel = make_dog_kernel(*kernel)
    return cv2.filter2D(frame.astype('float32'), -1, kernel)

def dog_kernel(sigma1, sigma2):
    """create a difference of gaussians kernel"""
    size = int(np.ceil(6 * sigma2))
    if size % 2 == 0:
        size += 1
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    r2 = x**2 + y**2
    g1 = np.exp(-r2 / (2 * sigma1**2)) / (2 * np.pi * sigma1**2)
    g2 = np.exp(-r2 / (2 * sigma2**2)) / (2 * np.pi * sigma2**2)
    dog = g1 - g2
    dog -= dog.mean()  
    return dog.astype(np.float32)


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
                     niter = 5000,
                     eps0 = 1e-10,
                     warp_mode = cv2.MOTION_EUCLIDEAN,
                     prepare = True,
                     gaussian_filter = 1,
                     conv_kernel = None,
                     hann = None,
                     **kwargs):
    h,w = template.shape
    if hann is None:
        hann = cv2.createHanningWindow((w,h),cv2.CV_32FC1)
        hann = (hann*255).astype('uint8')
    dst = frame.astype('float32')
    if not conv_kernel is None:
        dst = _conv_frame(dst,conv_kernel)

    if warp_mode in [cv2.MOTION_HOMOGRAPHY]: # added support for perspective transforms
        M = np.eye(3, 3, dtype=np.float32)
    else:
        M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                niter,  eps0)
    (res, M) = findTransformECC(template,dst,
                                M, warp_mode,
                                criteria,
                                inputMask=hann, gaussFiltSize=gaussian_filter)
    if M.shape[0] == 3:
        dst = cv2.warpPerspective(frame.astype('float32'), M, (w,h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    else:
        dst = cv2.warpAffine(frame.astype('float32'), M, (w,h),
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

def registration_upsample(frame,template,conv_kernel = None):

    h,w = frame.shape
    dst = frame.astype('float32')
    if not conv_kernel is None:
        dst = _conv_frame(dst,conv_kernel)
    (xs, ys), sf = cv2.phaseCorrelate(template.astype('float32'),dst)    
    M = np.float32([[1,0,xs],[0,1,ys]])
    dst = cv2.warpAffine(frame.astype('float32'),M,(w, h))
    return (xs,ys),(np.clip(dst,0,(2**16-1))).astype('uint16')

def _register_multichannel_stack(frames,templates,mode='2d',
                                 niter = 100,
                                 eps0 = 1e-3,
                                 warp_mode = cv2.MOTION_EUCLIDEAN,
                                 conv_kernel = None): # mode 2d

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
                         template = templates[ichan],conv_kernel= conv_kernel)
            ys[:,ichan] = np.array([r[0][1] for r in res],dtype='float32')
            xs[:,ichan] = np.array([r[0][0] for r in res],dtype='float32')

        elif mode == 'ecc':
            res = runpar(registration_ecc, chunk,
                         template = templates[ichan],
                         hann = hann,
                         niter = niter,
                         eps0 = eps0,
                         warp_mode = warp_mode,
                         conv_kernel = conv_kernel)
            xy,rots = _xy_rot_from_affine([r[0] for r in res])
            ys[:,ichan] = xy[:,1]
            xs[:,ichan] = xy[:,0]
            rot[:,ichan] = rots
        stack[:,ichan,:,:] = np.stack([r[1] for r in res])
    return (xs,ys,rot), stack

def motion_correct(dat, out = None,
                   refs = None,
                   chunksize=512,
                   nreference = 60,
                   mode = '2d',
                   diff_gaussians_filter = [3,5],
                   apply_shifts=True):
    '''
    Motion correction by translation.
    This estimate x and y shifts using phase correlation. 
    
    The reference image is the average of the chunk in the center.

    Inputs:
        dat (array)           : (NFRAMES, NCHANNEL, H, W) is overwritten if apply_shifts is True
        out (array)           : same size as dat or None to overwrite dat
        refs (array)          : reference frames (NCHANNEL, H, W) or None to compute from nreference frames
        chunksize (int)       : size of the chunks (needs to be small enough to fit in memory - default 512)
        nreference            : number of frames to take as reference (default 60)
        apply_shifts          : overwrite the data with the motion corrected (default True)
        mode                  : ecc (default) is rigid body; 2d is only translation in x and y using dft
        diff_gaussians_filter : convolve with a difference of gaussians to highlight the blood vessels (default [3,5], use None to skip)
    Returns:
        (yshifts, xshifts)    : shitfs in y and x ((NFRAMES, NCHANNELS),(NFRAMES, NCHANNELS))
        rot_shifts            : rotational shifts if in ecc mode 
    '''
    nframes,nchan,h,w = dat.shape
    conv_kernel = None
    if not diff_gaussians_filter is None:
        conv_kernel = dog_kernel(*diff_gaussians_filter)
    if out is None:
        out = dat
    chunks = chunk_indices(nframes,chunksize)
    xshifts = []
    yshifts = []
    rshifts = []
    # reference is from the start of the file (nreference frames to nreference*2)
    # (chunksize frames and for each channel independently)
    if refs is None:
        nreference = int(nreference)
        chunk = np.array(dat[nreference:nreference*2])
        refs = chunk[0].astype('float32')
        print(refs.shape)
        # align to the ref of each channel and use the mean
        _,refs = _register_multichannel_stack(chunk,refs,mode=mode,conv_kernel = conv_kernel)
        refs = np.mean(refs,axis=0).astype('float32')
    for c in tqdm(chunks,desc='Motion correction'):
        # this is the reg bit
        localchunk = np.array(dat[c[0]:c[-1]])
        (xs,ys,rot),corrected = _register_multichannel_stack(
            localchunk,
            refs,
            mode=mode,
            conv_kernel = conv_kernel)
        if apply_shifts:
            out[c[0]:c[-1]] = corrected[:]
            if hasattr(out,'flush'):
                out.flush() # write to disk
        yshifts.append(ys)
        xshifts.append(xs)
        rshifts.append(rot)
    return (np.vstack(yshifts),np.vstack(xshifts)),np.vstack(rshifts)


def _conv_frame(frame,kernel = None):
    '''
    Convolve frame with a kernel.
    does difference of gaussians if the type of kernel is list and len is 2
    returns float32
    '''
    if kernel is None:
        return frame
    if len(kernel) == 2 and type(kernel) is list:
        # then lets create the kernel
        kernel = make_dog_kernel(*kernel)
    return cv2.filter2D(frame.astype('float32'), -1, kernel)

def dog_kernel(sigma1, sigma2):
    """create a difference of gaussians kernel"""
    size = int(np.ceil(6 * sigma2))
    if size % 2 == 0:
        size += 1
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    r2 = x**2 + y**2
    g1 = np.exp(-r2 / (2 * sigma1**2)) / (2 * np.pi * sigma1**2)
    g2 = np.exp(-r2 / (2 * sigma2**2)) / (2 * np.pi * sigma2**2)
    dog = g1 - g2
    dog -= dog.mean()  
