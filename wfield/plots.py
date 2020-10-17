#  wfield - tools to analyse widefield data - plotting the results / summary plots 
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

from .utils import *
from .viz import imshow_noborder

def plot_summary_motion_correction(shifts, localdisk = None,
                                  title = 'Pixel shift (motion correction)'):
    '''
    Shifts are the saved motion correction pixels shifts. 
    shifts          : A rec array with the shifts per channel.
    localdisk       : Output folder (figure will be localdisk/figures/motion_correction.pdf)

    '''
    import pylab as plt
    fig = plt.figure(figsize=[9,5])

    fig.add_subplot(2,1,1)
    # need to make sure this doesnt crash with one channel
    nchannels = shifts['y'].shape[1]
    mi = shifts.view('float32').min() - 1
    ma = shifts.view('float32').max() + 1
    for i in range(nchannels):
        plt.plot(shifts['y'][:,i],label='channel {0}'.format(i+1))
    plt.title(title)
    plt.ylabel('y motion (pixels)')
    plt.legend()
    plt.ylim([mi,ma])

    fig.add_subplot(2,1,2)
    for i in range(nchannels):
        plt.plot(shifts['x'][:,i])
    plt.ylabel('x motion (pixels)')
    plt.xlabel('frame number');        
    plt.ylim([mi,ma])

    if not localdisk is None:
        folder = localdisk
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fig.savefig(pjoin(folder,'motion_correction.pdf'))
    return fig


def plot_summary_hemodynamics_dual_colors(rcoeffs,
                                          SVT_470,
                                          SVT_405,
                                          U,
                                          T,
                                          frame_rate=15.,
                                          duration = 6,
                                          outputdir=None):
    '''
    Approximates what was done in the hemodynamics correction.
    '''

    from skimage.filters import gaussian
    pidx = np.unravel_index(np.argmax(gaussian(rcoeffs,5)),
                           rcoeffs.shape) + np.array([0,-50])

    k,nframes = SVT_470.shape
    u = U[pidx[0]-2:pidx[0]+2,
          pidx[1]-2:pidx[1]+2,:].reshape([-1,k])
    mcoeff = np.mean(rcoeffs[pidx[0]-2:pidx[0]+2,
          pidx[1]-2:pidx[1]+2].reshape([-1]))
    # gets 60 seconds of data
    nframes = int(np.clip(duration*frame_rate,0,SVT_470.shape[1]))
    SVTa = SVT_470#SVT[:,0:nframes:2] 
    SVTb = SVT_405#SVT[:,1:nframes:2]
    # similar proprocessing as for correction
    SVTa = highpass(SVTa,.1,fs = frame_rate)
    SVTb = highpass(SVTb,.1,fs = frame_rate)
    # SVTb = lowpass(SVTb,10., fs = frame_rate)
    # SVTb = lowpass(SVTb,w = 15.,fs = frame_rate/2)
    SVTa = (SVTa.T - np.nanmean(SVTa,axis=1)).T.astype('float32')
    SVTb = (SVTb.T - np.nanmean(SVTb,axis=1)).T.astype('float32')
    SVTb_scaled = np.dot(T,SVTb)
    Ya = np.mean(u@SVTa[:,np.arange(1,nframes)],axis=0)
    Yb = np.mean(u@SVTb[:,np.arange(1,nframes)],axis=0)
    Ybscaled = np.mean(u@SVTb_scaled[:,np.arange(1,nframes)],axis=0)
    Ycorr =  np.mean(u@(SVTa[:,np.arange(1,nframes)]
                        - SVTb_scaled[:,np.arange(1,nframes)]),axis=0)
    
    import pylab as plt
    fig = plt.figure(figsize=[10,4])
    ax = fig.add_axes([.0,.0,.4,1])
    imshow_noborder(rcoeffs, fig = fig, 
               clim = np.percentile(rcoeffs,[20,98]),cmap='inferno')

    plt.colorbar(shrink = .4,label='regression coefficients')

    plt.axhline(y=pidx[0], color="w", linestyle="--",lw=1)
    plt.axvline(x=pidx[1], color="w", linestyle="--",lw=1)

    mi = np.min([np.nanmin(Ya),np.nanmin(Ybscaled)])*1.2
    ma = np.max([np.nanmax(Ya),np.nanmax(Ybscaled)])*1.2

    fig.add_axes([.4,0,.5,.5])
    plt.plot(Ya[:],color='#0066cc',label = 'blue')
    plt.plot(Yb[:],color='#cc00ff',label = 'violet')
    plt.plot(Ybscaled[:],lw = .5,color='k',label = 'scaled violet')
    plt.legend(loc='upper right')

    plt.ylim([mi,ma])
    plt.axis('off')
    fig.add_axes([.4,.5,.5,.5])
    plt.plot(Ycorr[:],'k',label = 'corrected')
    plt.legend(loc='upper right')
    plt.ylim([mi,ma])
    plt.axis('off')
    plt.plot([0,10*frame_rate/2],[mi,mi],'k',clip_on=False)
    plt.plot([0,0],[mi,mi+0.1],'k',clip_on=False)
    plt.text(5*frame_rate/2,mi,'10s',ha='center',va='bottom')
    plt.text(0,mi+0.05,'10% df/f',ha='right',va='center',rotation=90)
    if not outputdir is None:
        folder = outputdir
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fig.savefig(pjoin(folder,'hemodynamic_correction.pdf'))
    return fig
