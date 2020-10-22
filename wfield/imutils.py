from .utils import *

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
