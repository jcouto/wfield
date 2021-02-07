from .utils import *
from numpy.fft import fft
import math

def mask_to_3d(mask,shape,include_mask = None):
    '''
    3dmask = mask_to_3d(mask,shape,include_mask = None)
    
    Broadcasts a mask to a 3d array.

    Usage:
s1mask = mask_contour_on_extent(nrefregions[2]['c'],oriim.shape,extent=extent)[:,::-1]
v1mask = mask_contour_on_extent(nrefregions[6]['c'],oriim.shape,extent=extent)[:,::-1]
rlmask = mask_contour_on_extent(nrefregions[10]['c'],oriim.shape,extent=extent)[:,::-1]
s1moviemask = mask_to_3d([v1mask,s1mask,rlmask],tmp.shape,include_mask = winmask)
'''
    if type(mask) is list:
        mask = np.any(np.stack([m.astype(bool) for m in mask]),axis = 0)
    nmask = mask.copy().astype(mask.dtype)
    if not include_mask is None:
        nmask = np.all([nmask,include_mask],axis = 0)
    return np.broadcast_to(nmask,shape)

def get_signals_from_mask(dat, mask,include_mask = None):
    '''
    sigs = get_signals_from_mask(dat, mask,include_mask = None)

    Extract signals from a 2d or 3d array.
    
    Usage:
s1res = get_signals_from_mask(stimavgsf[0],s1mask,include_mask = winmask)
'''
    if type(mask) is list:
        mask = np.any(np.stack([m.astype(bool) for m in mask]),axis = 0)
    if not len(mask.shape) == len(dat.shape):
        mask = mask_to_3d(mask,dat.shape,include_mask = include_mask)
    res = dat[mask]
    if len(dat.shape) > 2:
        res = res.reshape((dat.shape[0],int(res.shape[0]/dat.shape[0])))
    return res

#############################################################
############Pseudo color and phase analysis##################
#############################################################

def hsv_colorbar_image():
    '''
    hsv_colorbar_image()
    
    
    Dumb way to make a colorbar:for an hsv image
    cbarim = hsv_colorbar_image()
    plt.imshow(cbarim,aspect= 'auto',extent = [0,1,-40,40])
    
    '''
    hsvimg = np.ones([1,255,3])*255
    hsvimg[:,:,0] = np.linspace(0,255,255)
    hsvimg = hsvimg.transpose([1,0,2])
    return cv2.cvtColor(hsvimg.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)

def fft_movie(movie, component = 1,output_raw = False):
    '''
    Computes the fft of a movie and returns the magnitude and phase 
    '''
    movief = fft(movie, axis = 0)
    if output_raw:
        return movief[component]
    phase  = -1. * np.angle(movief[component]) % (2*np.pi)
    mag = (np.abs(movief[component])*2.)/len(movie)
    return mag,phase

def fft_get_phase(movief):
    return -1. * np.angle(movief) % (2*np.pi)

def phasemap_to_visual_degrees(phasemap,startdeg,stopdeg):
    '''
    Normalizes the phasemap to visual angles
    Joao Couto 2019
    '''
    res = phasemap.copy() - np.nanmin(phasemap)
    res /= np.nanmax(res)
    res *= np.abs(np.diff([startdeg,stopdeg]))
    res += startdeg
    return res

def visual_sign_map(phasemap1, phasemap2):
    '''
    Computes the visual sign map from azimuth and elevation phase maps
    This is adapted from the Allen retinotopy code
    '''
    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)
    import scipy.ndimage as ni
    graddir1 = np.zeros(np.shape(gradmap1[0]))
    graddir2 = np.zeros(np.shape(gradmap2[0]))
    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])
    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))
    areamap = np.sin(np.angle(vdiff))
    return areamap

def im_fftphase_hsv(mov,component = 1,blur = 0,vperc=98,sperc=90,return_hsv=False):
    '''
    im_fftphase_hsv(mov,blur = 0,vperc=99,sperc=90)
    
    Creates a color image colorcoding the frame with fourier phase for each pixel 
    
        mov can be a 3d array or a list with the [magnitude, phase]
'''
    if not blur == 0:
        mov = runpar(im_gaussian,mov,sigma=blur)
    if not type(mov) is list:
        mag,H = fft_movie(mov,component = component)
    else:
        mag,H = mov
    H = H/(2*np.pi)
    V = mag.copy()
    V /= np.percentile(mag,vperc)
    S = mag**0.3
    S /= np.percentile(S,sperc)
    if return_hsv:
        return np.stack([H,S,V],axis=2).astype(np.float32)
    # Normalization for opencv ranges 0-255 for uint8
    hsvimg = np.clip(np.stack([H,S,V],axis=2).astype(np.float32),0,1)
    hsvimg *= 255
    return cv2.cvtColor(hsvimg.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)

def im_argmax_hsv(mov,blur = 0,vperc=98,sperc=90,return_hsv=False):
    '''
    im_argmax_hsv(mov,blur = 0,vperc=99,sperc=90)
    
    Creates a color image colorcoding the frame with largest amplitude for each pixel 
    '''
    if not blur == 0:
        mov = runpar(im_gaussian,mov,sigma=blur)
    H = np.argmax(mov,axis = 0)/np.float32(len(mov))
    maxim = np.max(mov,axis=0)
    meanim = np.mean(mov,axis=0)
    V = maxim.copy()
    V /= np.percentile(maxim,vperc)
    S = maxim-meanim
    S /= np.percentile(S,sperc)
    if return_hsv:
        return np.stack([H,S,V],axis=2).astype(np.float32)
    # Normalization for opencv ranges 0-255 for uint8
    hsvimg = np.clip(np.stack([H,S,V],axis=2).astype(np.float32),0,1)
    hsvimg *= 255
    return cv2.cvtColor(hsvimg.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)

def im_combineproj_hsv(stacks,
                       proj_funct = lambda x: np.nanstd(x,axis = 0),
                       vperc=98, sperc=98,
                       return_hsv=False):
    '''
    im_combineproj_hsv (stacks,
                        proj_funct=lambda x: np.std(x,axis = 0),
                        vperc=99, sperc=98)
    
    Creates a color image colorcoding the projections in space 
    '''
    proj = np.stack([proj_funct(x) for x in stacks])
    V,H = fft_movie(proj)
    H = H/(2*np.pi)
    S = V.copy()
    S = S
    S /= np.percentile(S,sperc)
    V /= np.percentile(V,vperc)
    if return_hsv:
        return np.stack([H,S,V],axis=2).astype(np.float32)
    # Normalization for opencv ranges 0-255 for uint8 
    hsvimg = np.clip(np.stack([H,S,V],axis=2),0,1)
    hsvimg[:,:,1:] *= 255
    hsvimg[:,:,0] *= 180
    return cv2.cvtColor(hsvimg.astype(np.uint8),
                        cv2.COLOR_HSV2RGB)                   

