# Motion correction
#
# This is mostly inspired/adapted from the amazing code of suite2p
# (by Carsen Stringer and Marius Pachitarius) and from
# skimage.register.phase_correlation 
#
# This is should be changed later, for widefield data one should probably add
# rigid body registration (translation + rotation)

from  .utils import *
from mkl_fft import fft2, ifft2
from numpy import fft
from numba import vectorize, complex64, float32, int16

def motion_correct(dat,chunksize=512, nreference = 60, apply_shifts=True):
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
    nframes,nchan,w,h = dat.shape

    mask = hamming_mask([w,h])
    chunks = chunk_indices(nframes,chunksize)
    xshifts = []
    yshifts = []
    # reference is from the middle of the file
    # (chunksize frames and for each channel independently)
    ichunk = int(len(chunks)/2)
    c = chunks[ichunk]
    chunk = dat[c[0]:c[0]+nreference].mean(axis = 0)
    maskmul = [0 for i in range(nchan)]
    maskoffset = [0 for i in range(nchan)]
    ref_phase = [0 for i in range(nchan)]
    for ichan in range(nchan): 
        ref_phase[ichan] = phasecorr_reference(chunk[ichan].astype('uint16'),
                                               mask = mask)
    chunk = np.array(dat[c[0]:c[0]+nreference])
    # align to the ref of each channel 
    for ichan in range(nchan): 
        y,x = phasecorr_shifts(chunk[:,ichan], ref_phase[ichan], mask = mask)
        shift_data(chunk[:,ichan],y,x)
        ref_phase[ichan] = phasecorr_reference(np.mean(chunk[:,ichan],axis=0).astype('uint16'),
                                               mask = mask)
    
    for c in tqdm(chunks,desc='Motion correction'):
        # this is the reg bit
        chunkdat = dat[c[0]:c[-1]]
        localchunk = np.array(chunkdat)
        ys = np.zeros((chunkdat.shape[0],chunkdat.shape[1],),dtype=int)
        xs = np.zeros((chunkdat.shape[0],chunkdat.shape[1],),dtype=int)
        for ichan in range(nchan):
            chunk = localchunk[:,ichan]
            y,x = phasecorr_shifts(chunk,
                                   ref_phase[ichan],
                                   mask)
            ys[:,ichan] = y
            xs[:,ichan] = x
        if apply_shifts:
            shift_data(chunkdat.reshape((-1,w,h)),
                       ys.reshape((-1)),xs.reshape((-1)))
        yshifts.append(ys)
        xshifts.append(xs)
    return np.vstack(yshifts),np.vstack(xshifts)

def hamming_mask(dims):
    '''
    This is used to mask the edges of the images.
    '''
    from scipy.signal.windows import hamming
    h,w = dims
    hh = hamming(h)
    ww = hamming(w)
    return np.sqrt(np.outer(hh,ww)).astype(np.float32)


def phasecorr_reference(img, mask = None):
    '''
    Prepare a reference for phase correlation.

    Multiplies by a mask first and normalizes by the abs of the fft.
    Returns the conjugate of an image

    Inputs:
    
    image (array)      : W by H image
    mask (array)       : W by H array that gets multiplied
    
    returns the reference image to use in phasecorr.
    '''
    if mask is None:
        mask = hamming_mask(img.shape)
    fft_ref = fft2(_apply_mask(img,mask))
    # normalize
    fft_ref /= np.abs(fft_ref) + 1e-5 #np.sqrt(np.mean(np.abs(fft_ref)**2)) + 1e-5
    return fft_ref.conjugate().astype('complex64')

def phasecorr_shifts(data, fft_ref, mask = None):
    '''
    Get translation using phase correlation.

    Inputs:
    
    data (array)       : NFRAMES by W by H image
    fft_ref (array)    : W by H array the conjugate of the normalized fft2 of the reference   
    mask (array)       : W by H array 
    
    returns the y and x shifts

    '''
    dims = data.shape[-2:]
    nframes = data.shape[0]
    if mask is None:
        # create a mask of none specified
        mask = hamming_mask(dims)
    # apply the mask and compute the fft
    X = _apply_mask(data, mask)
    for i in range(nframes):
        fft2(X[i],overwrite_x = True)
    # normalize the fft and multiply by the reference conj 
    X = _multiply_normalize(X, fft_ref)
    for i in range(nframes):
        ifft2(X[i],overwrite_x = True)
    
    amax = [np.unravel_index(np.argmax(np.real(x).astype('float32')),dims) for x in X]
    midpoints = np.array([np.fix(a / 2) for a in dims])
    shifts = np.array(amax, dtype=np.int32)
    for i,s in enumerate(shifts):
        shifts[i][s > midpoints] -= np.array(dims)[s > midpoints]
    return shifts.T

@vectorize(['complex64(uint16, float32)'],
           nopython=True, target = 'parallel')
def _apply_mask(x,mask):
    '''
    This trick gets a couple of seconds.
    https://github.com/MouseLand/suite2p 
    '''
    return np.complex64(x)*mask


@vectorize(['complex64(complex64, complex64)'],
           nopython=True, target = 'parallel')
def _multiply_normalize(target,reference):
    '''
    This trick  reduces time by half.
    https://github.com/MouseLand/suite2p
    '''
    nothing = np.complex64(1e-5)
    X = target / (nothing + np.abs(target))
    X = X*reference
    return X

def shift_data(X, ymax, xmax):
    """ rigid shift X by integer shifts ymax and xmax in place (no return)
        This is from suite2p 
    Parameters
    ----------
    X : int16
        array that's frames x Ly x Lx
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame

    """

    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]
    nimg, Ly, Lx = X.shape
    for n in range(nimg):
        X[n] = np.roll(X[n].copy(), (-ymax[n], -xmax[n]), axis=(0,1))
