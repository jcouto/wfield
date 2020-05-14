from numba import jit
from numba import int16 as numba_int16

@jit(nopython=True)
def allen_top_proj_from_volume(bvol):
    '''
Get the top projection from a volume.

    proj = allen_top_proj_from_volume(allen_volume)

'''
    h,d,w = bvol.shape
    proj = np.zeros((h,w),dtype=numba_int16)
    # this can be done with a np.where but is probably faster like this
    for i in range(h):
        for j in range(w):
            for z in range(d): 
                if bvol[i,z,j] > 0:
                    proj[i,j] = bvol[i,z,j]
                    break
            
    return proj
