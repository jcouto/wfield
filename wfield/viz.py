#  wfield - tools to analyse widefield data - visualization 
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

def imshow_noborder(img,fig = None,figsize = [7,7],**kwargs):
    if fig is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes([0,0,1,1])
    plt.gca().axis('off')            
    plt.imshow(img,**kwargs)
    return fig

###############################################################
#######################  NOTEBOOK WRAPPERS  ###################
###############################################################

def _handle_sparse(im,shape):
    if issparse(im):
        if shape is None:
            raise ValueError('Supply shape = [H,W] when using sparse arrays')
        im = np.asarray(im.todense()).reshape(shape)
    return im

def nb_play_movie(data,interval=30,shape = None,**kwargs):
    ''' 
    Play a movie from the notebook
    '''
    from ipywidgets import Play,jslink,HBox,IntSlider
    from IPython.display import display

    i = _handle_sparse(data[0],shape = shape)
    im = plt.imshow(i.squeeze(),**kwargs)
    slider = IntSlider(0,min = 0,max = data.shape[0]-1,step = 1,description='Frame')
    play = Play(interval=interval,
                value=0,
                min=0,
                max=data.shape[0]-1,
                step=1,
                description="Press play",
                disabled=False)
    jslink((play, 'value'), (slider, 'value'))
    display(HBox([play, slider]))
    def updateImage(change):
        i = _handle_sparse(data[change['new']],shape=shape)
        im.set_data(i.squeeze())
    slider.observe(updateImage, names='value')
    return dict(fig = plt.gcf(),
                ax=plt.gca(),
                im= im,
                update = updateImage)

def nb_save_movie(data,filename,interval = 100,dpi = 90,shape=None,**kwargs):
    '''
    Replace nb_play_movie with this to save to a file.

    Example:

    nb_save_movie(tmp[:,:,::-1],
                  filename = '~/Desktop/example.avi',
                  clim = [.06,.2],
                  extent = extent,
                  cmap = 'hot', 
                  alpha = 0.5);
    '''
    from tqdm import tqdm
    from matplotlib.animation import FuncAnimation
    def animate(frame):
        global pbar
        pbar.update(1)
        i = _handle_sparse(data[frame],shape = shape)
        im.set_data(i.squeeze())        
        return im,
    fig = plt.gcf()
    i = _handle_sparse(data[0],shape = shape)
    im = plt.imshow(i.squeeze(),**kwargs)
    animation = FuncAnimation(
        fig,
        animate,
        np.arange(data.shape[0]),
        fargs=[],
        interval=interval)
    global pbar
    pbar = tqdm(desc = 'Saving movie ',total=data.shape[0])
    animation.save(filename, dpi=dpi)
    pbar.close()
    plt.show()
    print('Saved to {0}'.format(filename))



################################################################
##################### PYQTGRAPH WRAPPERS #######################
################################################################

def qtgraph_show_svd(stack):
    from .widgets import SVDViewer
    return SVDViewer(stack)

################################################################
##################### HOLOVIEWS WRAPPERS #######################
################################################################

def hv_imshow_stack(X,cmap = 'gray',scale=.8,title='dataset',timelabel = 'frame'):
    import holoviews as hv
    hv.extension('bokeh')
    d = X.shape
    if len(d) == 4:
        ds = hv.Dataset((np.arange(d[3]), np.arange(d[2]), np.arange(d[1]),np.arange(d[0]),
                         X),
                        ['h','w', 'ch', timelabel], title)
    elif len(d) == 3:
        ds = hv.Dataset((np.arange(d[2]), np.arange(d[1]),np.arange(d[0]),
                         X),
                        ['h','w', timelabel], title)
    else:
        print('imshow_stack only works for 3d and 4d stacks.')
        return None
    im = ds.to(hv.Image, ['h', 'w'],dynamic=True)
    im.opts(cmap=cmap,width=int(d[-1]*scale),height=int(d[-2]*scale))
    return im

################################################################
#######################  NAPARI WRAPPERS #######################
################################################################


def napari_show(dat,contrast_limits = None):
    import napari
    if contrast_limits is None:
        contrast_limits = [dat[0].min(),dat[0].max()]
    with napari.gui_qt():
        try:
            napari.view_image(dat,
                              contrast_limits=contrast_limits,
                              is_pyramid=False)
        except: # napari 0.3.5 .... why??
            napari.view_image(dat,
                              contrast_limits=contrast_limits)
            
