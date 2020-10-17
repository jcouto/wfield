#  wfield - tools to analyse widefield data - IO tools 
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

def load_dat(filename,
             nframes = None,
             offset = 0,
             shape = None,
             dtype='uint16'): 
    '''
    Loads frames from a binary file.
    
    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16) 
    Returns:
        An array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = load_dat(filename)
        
    ''' 
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None: # try to get it from the filename
        meta = os.path.splitext(filename)[0].split('_')
    if dtype is None:
        dtype = meta[-1]
        dt = np.dtype(dtype)
    if shape is None:
        shape = [int(m) for m in meta[-4:-1]]
    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    framesize = int(np.prod(shape))
    dt = np.dtype(dtype)
    offset = int(offset)
    with open(filename,'rb') as fd:
        fd.seek(offset*framesize*int(dt.itemsize))
        buf = np.fromfile(fd,dtype = dt, count=framesize*nframes)
    buf = buf.reshape((-1,*shape),
                      order='C')
    return buf

def mmap_dat(filename,
             mode = 'r',
             nframes = None,
             shape = None,
             dtype='uint16'):
    '''
    Loads frames from a binary file as a memory map.
    This is useful when the data does not fit to memory.
    
    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        mode (str)           : memory map access mode (default 'r')
                'r'   | Open existing file for reading only.
                'r+'  | Open existing file for reading and writing.                 
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16) 
    Returns:
        A memory mapped  array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = mmap_dat(filename)
    '''
    
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None: # try to get it from the filename
        meta = os.path.splitext(filename)[0].split('_')
        if shape is None:
            shape = [int(m) for m in meta[-4:-1]]
        if dtype is None:
            dtype = meta[-1]
    dt = np.dtype(dtype)
    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    dt = np.dtype(dtype)
    return np.memmap(filename,
                     mode=mode,
                     dtype=dt,
                     shape = (int(nframes),*shape))

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
    dd = load_binary_block((filename,trial_onset,nbaseline_frames),
                           shape=shape,dtype=dtype)
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
    from skvideo.io import FFmpegReader
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
    stim_onset,stim_offset = analog_ttl_to_onsets(dat[-4,:],time=None)
    ch1,ch1_ = analog_ttl_to_onsets(dat[-2,:],time=None)
    ch2,ch2_ = analog_ttl_to_onsets(dat[-1,:],time=None)
    info = dict(baseline = header['baseline'],
                ch1 = ch1,
                ch2 = ch2,
                stim_onset = stim_onset,
                stim_offset = stim_offset,
                onset = header['onset'])
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
                            chunksize = 1,
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
    frametrial = [[0,0,0]]
    framesinfo = []
    cnt_trials = 0
    frames_avg = np.zeros((nchannels,w,h),dtype=float).squeeze()
    fnames_mj2_chunks = fnames_mj2
    if chunksize > 1: 
        fnames_mj2_chunks = [fnames_mj2[a:b] for a,b in chunk_indices(len(fnames_mj2),chunksize)]
    with open(dat_path,'wb') as dat:
        for itrial,f in tqdm(enumerate(fnames_mj2_chunks),
                             total=len(fnames_mj2_chunks),
                             desc='Collecting imager trials'):
            if chunksize > 1: 
                res = runpar(split_imager_channels,f)
            else:
                res = [split_imager_channels(f)]
            for ch1,ch2,info in res:
                cnt_trials += 1
                if ch1 is None:
                    print('Skipped trial {0} (no frames).'.format(f))
                    continue
                # compute the stimuli onsets
                if not len(info['stim_onset']) and not len(info['stim_offset']):
                    print('Trial {0} had no stimulus; skiped trial.'.format(cnt_trials))
                    continue
                elif not len(info['stim_onset']):
                    stim_onset_frames = 0 # if there is was an offset but no onset, the stim onset is the first frame (this happens on the visual stim case...) 
                else:
                    try:
                        stim_onset_frames = np.where(info['ch1']<info['stim_onset'])[0][-1]
                    except:
                        print('There was an error in the imager: {0}'.format(info['stim_onset']))
                        stim_onset_frames = 0 # if there is was an offset but no onset, the stim onset is the first frame (this happens on the visual stim case...) 
                d = np.stack([ch1,ch2]).transpose([1,0,2,3])
                frames_avg += np.mean(d,axis = 0).squeeze()
                d = d.reshape([-1,w,h])
                lenframes = frametrial[-1][1] +len(d)/2
                # write meta and frames
                frametrial[-1][2] = frametrial[-1][1] + stim_onset_frames
                frametrial.append([cnt_trials,lenframes,lenframes])
                dat.write(d.astype(dtype))
                framesinfo.append(dict(itrial=cnt_trials,**info))
    frames_avg /= float(len(frametrial))
    np.save(pjoin(destination,'frames_average.npy'),
            frames_avg.astype(np.float32))
    tstop = time.time()
    print('Took {0} min to collect data and compute the averages'.format((tstop-tstart)/60))
    # Save trial onset frames
    trialonsets = np.array(frametrial)
    np.save(pjoin(destination,'trial_onsets.npy'),trialonsets[:-1])
    # Save trial information
    trialinfo = pd.DataFrame(framesinfo)
    trialinfo.to_csv(pjoin(destination,'trial_info.csv'))
    return mmap_dat(dat_path), frames_avg, trialonsets,trialinfo

