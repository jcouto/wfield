from .utils import *
import cv2
from skvideo.io import FFmpegReader

def mmap_dat(fname,
             mode = 'r',
             nchan = None,
             w=None,
             h=None,
             dtype=None,
             nframes=None):
    assert os.path.isfile(fname)
    meta = os.path.splitext(fname)[0].split('_')
    if nchan is None: # try to get it from the filename
        nchan = int(meta[-4])
    if w is None: # try to get it from the filename
        w = int(meta[-3])
    if h is None: # try to get it from the filename
        h = int(meta[-2])
    if dtype is None: # try to get it from the filename
        dtype = meta[-1]
    dt = np.dtype(dtype)
    if nframes is None:
        # Get the number of samples from the file size
        nsamples = os.path.getsize(fname)/(nchan*w*h*dt.itemsize)
    return np.memmap(fname,
                     mode=mode,
                     dtype=dt,
                     shape = (int(nsamples),int(nchan),int(w),int(h)))

def load_binary_block(block, #(filename,onset,offset)
                       shape = None,
                      dtype='uint16'): 
    '''Loads a block from a binary file (nchannels,W,H)'''
    fname,offset,bsize = block
    nchans,W,H = shape
    framesize = int(nchans*W*H)
    dt = np.dtype(dtype)
    offset = int(offset)
    with open(fname,'rb') as fd:
        fd.seek(offset*framesize*int(dt.itemsize))
        buf = np.fromfile(fd,dtype = dt, count=framesize*bsize)
    buf = buf.reshape((-1,nchans,W,H),
                               order='C')
    return buf

def compute_trial_baseline_from_binary(trial_onset,
                                       filename,
                                       shape,
                                       nbaseline_frames,
                                       dtype='uint16'):
    dd = load_binary_block((filename,trial_onset,nbaseline_frames),shape=shape,dtype=dtype)
    return dd.mean(axis=0)

def frames_average_for_trials(dat,onsets,nbaseline_frames):
    from .utils import runpar
    if hasattr(dat,'filename'):
        dims = dims = dat.shape[1:]
        dat_path = dat.filename
        frames_average = runpar(compute_trial_baseline_from_binary,onsets,
                                filename = dat_path,
                                shape=dims,
                                nbaseline_frames=nbaseline_frames,
                                dtype = dat.dtype)
    else:
        frame_averages = []
        for on in tqdm(onsets):
            frame_averages.append(dat[on:on+nbaseline_frames].mean(axis=0))
    return np.stack(frames_average)

#######################################################################
################        PARSE WIDEFIELDIMAGER       ###################
#######################################################################

def read_mj2_frames(fname):
    sq = FFmpegReader(fname,outputdict={'-pix_fmt':'gray16le'})
    imgs = []
    for s in sq:
        imgs.append(s)
    sq.close()
    return np.stack(imgs).squeeze()

def read_imager_analog(fname):
    from struct import unpack
    with open(fname,'rb') as fd:
        tstamp = unpack("d", fd.read(8))[0]
        onset = unpack("d", fd.read(8))[0]
        nchannels = int(unpack("<d", fd.read(8))[0])
        nsamples = unpack("<d", fd.read(8))[0]
        dat = np.fromfile(fd,dtype='uint16')
        dat = dat.reshape((-1,nchannels)).T
    return dat,dict(baseline=tstamp,
                    onset=onset,
                    nchannels=nchannels,
                    nsamples=nsamples)

def split_imager_channels(fname_mj2):
    ''' splits channels from the imager '''
    from .utils import analog_ttl_to_onsets
    fname_analog = fname_mj2.replace('Frames_','Analog_').replace('.mj2','.dat')
    stack = read_mj2_frames(fname_mj2)
    dat,header = read_imager_analog(fname_analog)
    info = dict(baseline = header['baseline'],
                onset = header['onset'],
                ch2 = analog_ttl_to_onsets(dat[-1,:],time=None),
                ch1 = analog_ttl_to_onsets(dat[-2,:],time=None),
                stim_onset = analog_ttl_to_onsets(dat[-4,:],time=None))
    nframes = stack.shape[0]
    avgnorm = stack.reshape((nframes,-1))
    avgnorm = avgnorm.mean(axis=1)
    avgnorm -= np.mean(avgnorm)/2
    # find the last frame with the LED on; that's ch0
    idx = np.where(avgnorm>0)[0][-1]
    # Check if the last frame is odd or even
    if idx & 1:
        chAidx = np.arange(nframes)[1::2]
        chBidx = np.arange(nframes)[0::2]
    else:
        chAidx = np.arange(nframes)[0::2]
        chBidx = np.arange(nframes)[1::2]
    # drop empty frames
    chAidx = chAidx[avgnorm[chAidx]>0]
    chBidx = chBidx[avgnorm[chBidx]>0]
    if not len(info['ch1']) or not len(info['ch2']):
        print('Could not parse {0}'.format(fname_mj2))
        print(info)
        return None,None,info
    # look at the analog channels to know which was the last channel.
    lastch = np.argmax([info['ch1'][-1],info['ch2'][-1]])
    if lastch == 1: # last channel was channel 2
        ch1idx = chBidx
        ch2idx = chAidx
    else:
        ch1idx = chAidx
        ch2idx = chBidx
    # collect an equal number of frames for both channels, ch1 first
    if ch1idx[0] > ch2idx[0]:
        # drop first ch2
        ch2idx = ch2idx[1:]
        info['ch2'] = info['ch2'][1:]         
    if ch1idx[-1] > ch2idx[-1]:
        # drop last ch1
        ch1idx = ch1idx[:-1]
        info['ch1'] = info['ch1'][:-1] 
    return stack[ch1idx],stack[ch2idx],info

def parse_imager_mj2_folder(folder, destination, 
                            nchannels = 2,
                            dtype = 'uint16'):
    fnames_mj2 = natsorted(glob(pjoin(folder,'Frames_*.mj2')))
    if not len(fnames_mj2):
        print('Folder empty or not accessible {0}'.format(folder))
    sample = read_mj2_frames(fnames_mj2[0])
    nframes,w,h = sample.shape
    
    # cast to float32 so that we can subract the mean (this is sort of dumb..)
    dat_path = pjoin(destination,'frames_{0}_{1}_{2}_{3}.dat'.format(nchannels,w,h,dtype))

    if not os.path.isdir(destination):
        print('Creating output directory {0}'.format(destination))
        os.makedirs(destination)

    tstart = time.time()
    frametrial = [(0,0)]
    framesinfo = []
    cnt_trials = 0
    frames_avg = np.zeros((nchannels,w,h),dtype=float).squeeze()
    with open(dat_path,'wb') as dat:
        for itrial,f in tqdm(enumerate(fnames_mj2),
                             total=len(fnames_mj2),
                             desc='Collecting imager trials'):
            ch1,ch2,info = split_imager_channels(f)
            if ch1 is None:
                print('Skipped trial {0}'.format(f))
                continue
            d = np.stack([ch1,ch2]).transpose([1,0,2,3])
            frames_avg += np.mean(d,axis = 0).squeeze()
            d = d.reshape([-1,w,h])
            frametrial.append((itrial,frametrial[-1][1]+len(d)/2))
            dat.write(d.astype(dtype))
            framesinfo.append(dict(itrial=itrial,**info))
            cnt_trials += 1
    frames_avg /= float(cnt_trials)
    np.save(pjoin(destination,'frames_average.npy'),
            frames_avg.astype(np.float32))
    tstop = time.time()
    print('Took {0} min to collect data and compute the averages'.format((tstop-tstart)/60))
    # Save trial onset frames
    trialonsets = np.rec.array(frametrial,
                              dtype=([('itrial','i4'),('iframe','i4')]))
    np.save(pjoin(destination,'trial_onsets.npy'),trialonsets[:-1])
    # Save trial information
    trialinfo = pd.DataFrame(framesinfo)
    trialinfo.to_csv(pjoin(destination,'trial_info.csv'))
    return mmap_dat(dat_path), frames_avg, trialonsets,trialinfo

