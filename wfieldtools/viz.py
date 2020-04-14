import pylab as plt
from .utils import *

def two_chan_to_rgb(dat,norm=True):
    img = np.stack([*dat,
                          np.zeros_like(dat[0])],
                   axis = 0)
    if norm:
        for i in range(len(dat)):
            img[i] /= img[i].max()
        img *= 255
        img = img.astype('uint8')
    img = img.transpose([1,2,0])
    return img


################################################################
#######################  MATPLOTLIB WRAPPERS ###################
################################################################

def imshow_noborder(img,figsize = [7,7],**kwargs):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    plt.imshow(img,**kwargs)
    return fig,ax


################################################################
#######################  NAPARI WRAPPERS #######################
################################################################


def napari_show(dat,contrast_limits = None):
    import napari
    if contrast_limits is None:
        contrast_limits = [dat[0].min(),dat[0].max()]
    with napari.gui_qt():
        napari.view_image(dat,
                          contrast_limits=contrast_limits,
                          is_pyramid=False)

