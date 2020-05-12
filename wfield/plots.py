# Functions for plotting the results / summary plots
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
    mi = shifts.view('int').min() - 1
    ma = shifts.view('int').max() + 1
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
        folder = pjoin(localdisk,'figures')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fig.savefig(pjoin(folder,'motion_correction.pdf'))
    return fig


def plot_summary_hemodynamics_dual_colors(rcoeffs,
                                          SVT,
                                          U,
                                          T,
                                          frame_rate=30.,
                                          duration_frames = 60,
                                          outputdir=None):
    '''
    Approximates what was done in the hemodynamics correction.
    '''

    from skimage.filters import gaussian
    pidx = np.unravel_index(np.argmax(gaussian(rcoeffs,5)),
                           rcoeffs.shape) + np.array([0,-50])

    k,nframes = SVT.shape
    u = U[pidx[0]-2:pidx[0]+2,
          pidx[1]-2:pidx[1]+2,:].reshape([-1,k])
    mcoeff = np.mean(rcoeffs[pidx[0]-2:pidx[0]+2,
          pidx[1]-2:pidx[1]+2].reshape([-1]))
    # gets 60 seconds of data
    nframes = int(np.clip(duration_frames*frame_rate,0,SVT.shape[1]))
    SVTa = SVT[:,-nframes::2] 
    SVTb = SVT[:,-nframes::2]
    # similar proprocessing as for correction
    VTa = highpass(SVTa,.1,fs = frame_rate/2)
    SVTb = highpass(SVTb,.1,fs = frame_rate/2)
    # SVTb = lowpass(SVTb,w = 15.,fs = frame_rate/2)
    SVTa = (SVTa.T - np.nanmean(SVTa,axis=1)).T.astype('float32')
    SVTb = (SVTb.T - np.nanmean(SVTb,axis=1)).T.astype('float32')
    SVTb_scaled = np.dot(T.T,SVTb)
    Ya = np.mean(u@SVTa,axis=0)
    Yb = np.mean(u@SVTb,axis=0)
    Ybscaled = np.mean(u@SVTb_scaled,axis=0)
    Ycorr =  np.mean(u@(SVTa-SVTb_scaled),axis=0)
    
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
        folder = pjoin(outputdir,'figures')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fig.savefig(pjoin(folder,'hemodynamic_correction.pdf'))
    return fig