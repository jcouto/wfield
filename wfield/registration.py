# Motion registration code
# This is mostly adapted from code in suite2p by Carsen Stringer and Marius Pachitarius 
from  .utils import *
from mkl_fft import fft2, ifft2
from numpy import fft
from numba import vectorize, complex64, float32, int16

def motion_correct(dat,chunksize=500, mask_edge = 100, apply_shifts=True):
    avgdat = np.zeros(dat.shape[1:],dtype=np.float32)
    nframes,nchan,w,h = dat.shape
    chunks = np.array_split(np.arange(nframes),int(nframes/chunksize))
    nchunks = np.float32(len(chunks))
    xshifts = []
    yshifts = []
    # reference is from the middle of the file
    # (chunksize frames and for each channel independently)
    ichunk = int(len(chunks)/2) 
    chunk = dat[chunks[ichunk][0]]
    maskmul = [0 for i in range(nchan)]
    maskoffset = [0 for i in range(nchan)]
    ref_phase = [0 for i in range(nchan)]
    (maskmul[0],maskoffset[0],ref_phase[0]) = phasecorr_reference(
        chunk[1], mask_edge = mask_edge)
    chunk = np.array(dat[chunks[0][0]:chunks[0][-1]])
    # align to the ref of channel 1
    for ichan in range(nchan): 
        y,x,_ = phasecorr(chunk[:,ichan],maskmul[0],maskoffset[0],ref_phase[0])
        shift_data(chunk[:,ichan],y,x)
        (maskmul[ichan],maskoffset[ichan],
         ref_phase[ichan]) = phasecorr_reference(np.mean(chunk[:,ichan],axis=0),
                                                 mask_edge = mask_edge)
    for c in tqdm(chunks,desc='Motion correction'):
        # this is the reg bit
        chunkdat = dat[c[0]:c[-1]]
        localchunk = np.array(chunkdat)
        ys = np.zeros((chunkdat.shape[0],chunkdat.shape[1],),dtype=int)
        xs = np.zeros((chunkdat.shape[0],chunkdat.shape[1],),dtype=int)
        for ichan in range(nchan):
            chunk = localchunk[:,ichan]
            y,x,_ = phasecorr(chunk,maskmul[ichan],
                              maskoffset[ichan],
                              ref_phase[ichan])
            ys[:,ichan] = y
            xs[:,ichan] = x
        if apply_shifts:
            shift_data(chunkdat.reshape((-1,w,h)),
                       ys.reshape((-1)),xs.reshape((-1)))
        yshifts.append(ys)
        xshifts.append(xs)
        #this is the average but
        avgdat += np.mean(chunkdat,axis=0)/nchunks
    return np.vstack(yshifts),np.vstack(xshifts),avgdat

def phasecorr_reference(img,mask_edge = 50,gaussian_sigma = 0):
    '''This is adapted from suite2p '''
    w,h = img.shape
    
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
    '''
    Adapted from suite2p 
    '''
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
    ''' 
    Gaussian filter in the fft domain with std sig and size Ly,Lx
    This is from suite2p 
    '''
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
    ''' spatial taper  on edges with gaussian of std sig 
        This is from suite2p '''
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
def dotnorm(Y, ref):
    ''' This is from suite2p '''
    eps0 = np.complex64(1e-5)
    x = Y / (eps0 + np.abs(Y))
    x = x*ref
    return x

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
