# Motion registration
#
# This is mostly inspired/adapted from code in suite2p (by Carsen Stringer
# and Marius Pachitarius) and from skimage.register.phase_correlation 
#

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
        mask_edge             : how much to mask the edges of the image (default: 50)
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
        ref_phase[ichan] = phasecorr_reference(chunk[ichan],
                                               mask = mask)
    chunk = np.array(dat[c[0]:c[0]+nreference])
    # align to the ref of each channel 
    for ichan in range(nchan): 
        y,x = phasecorr_shifts(chunk[:,ichan], ref_phase[ichan], mask = mask)
        shift_data(chunk[:,ichan],y,x)
        ref_phase[ichan] = phasecorr_reference(np.mean(chunk[:,ichan],axis=0),
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
    fft_ref = fft2(img.astype('float32')*mask)
    # normalize
    fft_ref /= np.abs(fft_ref) + 1e-5 #np.sqrt(np.mean(np.abs(fft_ref)**2)) + 1e-5
    return fft_ref.conjugate().astype('complex64')

def phasecorr_shifts(data, fft_ref, mask = None):
    dims = data.shape[-2:]
    if mask is None:
        mask = hamming_mask(dims)
        
    X = fft2(data.astype('float32')*mask)
    #tstart = time.time()
    #norm = np.sqrt(np.mean((np.abs(X)**2).reshape(
    #    X.shape[0],-1),axis = 1)) + 1e-5
    #X = (X.T / norm).T
    #print(time.time()-tstart)

    X /= np.abs(X) + 1e-5
    X = ifft2(X*fft_ref)
    amax = [np.unravel_index(np.argmax(np.real(x).astype('float32')),dims) for x in X]
    midpoints = np.array([np.fix(a / 2) for a in dims])
    shifts = np.array(amax, dtype=np.int32)
    for i,s in enumerate(shifts):
        shifts[i][s > midpoints] -= np.array(dims)[s > midpoints]
    return shifts.T

'''
def phasecorr_reference(img,mask_edge = 50,gaussian_sigma = 0):
    
    maskmul = spatial_taper(mask_edge, w, h).astype('float32')

    maskoffset = img.mean() * (1. - maskmul)
    
    fft_ref = np.conj(fft2(img.copy()))
    abs_ref = np.absolute(fft_ref)
    ref_img = fft_ref / (1e-5 + abs_ref)
    if gaussian_sigma:
        fhg = gaussian_fft(gaussian_sigma, w, h)
        ref_img *= fhg
                  
    return (maskmul.astype('float32'),
            maskoffset.astype('float32'),
            ref_img.astype('complex64'))

def phasecorr(data, maskmul, maskoffset, ref_phase,lcorr = 30):   
    nimg = data.shape[0]
    ly,lx = ref_phase.shape[-2:]
    lyhalf = int(np.floor(ly/2))
    lxhalf = int(np.floor(lx/2))

    # shifts and corrmax
    ymax = np.zeros((nimg,), np.int32)
    xmax = np.zeros((nimg,), np.int32)
    cmax = np.zeros((nimg,), np.float32)

    X = addmultiplytype(data.copy(), maskmul, maskoffset)
    
    for t in range(X.shape[0]):
        fft2(X[t], overwrite_x=True)

    X = dotnorm(X, ref_phase)
    for t in np.arange(nimg):
        ifft2(X[t], overwrite_x=True)
    x00, x01, x10, x11 = shift_crop(X, lcorr)
    x00,x01,x10,x11
    cc = np.real(np.block([[x11, x10], [x01, x00]]))

    for t in np.arange(nimg):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None),
                                            (2*lcorr+1, 2*lcorr+1))
        cmax[t] = cc[t, ymax[t], xmax[t]]
    ymax, xmax = ymax-lcorr, xmax-lcorr
    return ymax,xmax,cmax


def gaussian_fft(sig, Ly, Lx):
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    return fhg

def spatial_taper(sig, Ly, Lx):
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2*sig
    mX = x.max() - 2*sig
    maskY = 1./(1.+np.exp((yy-mY)/sig))
    maskX = 1./(1.+np.exp((xx-mX)/sig))
    maskMul = maskY * maskX
    return maskMul

@vectorize([complex64(complex64, complex64)], 
           nopython=True,
           target = 'parallel')

@vectorize(['complex64(uint16, float32, float32)',
            'complex64(float32, float32, float32)'],
           nopython=True, target = 'parallel')
def addmultiplytype(x,y,z):
    return np.complex64(np.float32(x)*y + z)


def shift_crop(X, lcorr):
    """ 
    Perform 2D fftshift and crop with lcorr
    This is from suite2p 
    """
    x00 = X[:,  :lcorr+1, :lcorr+1]
    x11 = X[:,  -lcorr:, -lcorr:]
    x01 = X[:,  :lcorr+1, -lcorr:]
    x10 = X[:,  -lcorr:, :lcorr+1]
    return x00, x01, x10, x11

'''

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
