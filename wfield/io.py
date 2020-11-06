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
        dtype,shape,_ = _parse_binary_fname(filename,
                                            shape = shape,
                                            dtype = dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype

    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    framesize = int(np.prod(shape))

    offset = int(offset)
    with open(filename,'rb') as fd:
        fd.seek(offset*framesize*int(dt.itemsize))
        buf = np.fromfile(fd,dtype = dt, count=framesize*nframes)
    buf = buf.reshape((-1,*shape),
                      order='C')
    return buf

def _parse_binary_fname(fname,lastidx=None, dtype = 'uint16', shape = None, sep = '_'):
    '''
    Gets the data type and the shape from the filename 
    This is a helper function to use in load_dat.
    
    out = _parse_binary_fname(fname)
    
    With out default to: 
        out = dict(dtype=dtype, shape = shape, fnum = None)
    '''
    fn = os.path.splitext(os.path.basename(fname))[0]
    fnsplit = fn.split(sep)
    fnum = None
    if lastidx is None:
        # find the datatype first (that is the first dtype string from last)
        lastidx = -1
        idx = np.where([not f.isnumeric() for f in fnsplit])[0]
        for i in idx[::-1]:
            try:
                dtype = np.dtype(fnsplit[i])
                lastidx = i
            except TypeError:
                pass
    if dtype is None:
        dtype = np.dtype(fnsplit[lastidx])
    # further split in those before and after lastidx
    before = [f for f in fnsplit[:lastidx] if f.isdigit()]
    after = [f for f in fnsplit[lastidx:] if f.isdigit()]
    if shape is None:
        # then the shape are the last 3
        shape = [int(t) for t in before[-3:]]
    if len(after)>0:
        fnum = [int(t) for t in after]
    return dtype,shape,fnum


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
        dtype,shape,_ = _parse_binary_fname(filename,
                                            shape = shape,
                                            dtype = dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype
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

def _imager_parse_file(fname, version = 2):
    f,ext = os.path.splitext(fname)
    if ext == '.mj2':
        stack = read_mj2_frames(fname)
    else:
        stack = mmap_dat(fname)
    
    stack = stack.reshape([-1,*stack.shape[-2:]])
    stack = stack[:,100,:]
    stacksize = len(stack)
    _,_,fnum = _parse_binary_fname(fname)
    folder = os.path.dirname(fname)
    analog,analogheader = read_imager_analog(pjoin(folder,'Analog_{0}.dat'.format(fnum[0])))
    idxch1,idxch2,info = _imager_split_channels(stack,analog,analogheader,version = version)
    info['nrecorded'] = len(stack)
    del stack
    return idxch1,idxch2,info

def _imager_split_channels(stack,analog,analogheader,version = 2):
    ''' splits channels from the imager '''
    from .utils import analog_ttl_to_onsets
    #fname_analog = fname_mj2.replace('Frames_','Analog_').replace('.mj2','.dat')
    #stack = read_mj2_frames(fname_mj2)
    #dat,header = read_imager_analog(fname_analog)

    if version == 1:
        # then the channels are -2 and -1
        ch1,ch1_ = analog_ttl_to_onsets(analog[-1,:],time=None) # this is the blue LED
        ch2,ch2_ = analog_ttl_to_onsets(analog[-2,:],time=None) # this is the violet LED
        ch3_onset,ch3_offset = analog_ttl_to_onsets(analog[3,:],time=None) # this is the stim
        ch4_onset,ch4_offset = analog_ttl_to_onsets(analog[4,:],time=None) # this is another sync
    else:
        ch1,ch1_ = analog_ttl_to_onsets(analog[1,:],time=None) # this is the blue LED
        ch2,ch2_ = analog_ttl_to_onsets(analog[2,:],time=None) # this is the violet LED
        ch3_onset,ch3_offset = analog_ttl_to_onsets(analog[3,:],time=None) # this is the stim
        ch4_onset,ch4_offset = analog_ttl_to_onsets(analog[4,:],time=None) # this is another sync
    
    info = dict(baseline = analogheader['baseline'],
                ch1 = ch1,
                ch2 = ch2,
                ch3_onset = ch3_onset,
                ch3_offset = ch3_offset,
                ch4_onset = ch4_onset,
                ch4_offset = ch4_offset,
                onset = analogheader['onset'])
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
    return ch1idx,ch2idx,info

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
    from pandas import DataFrame
    trialinfo = DataFrame(framesinfo)
    trialinfo.to_csv(pjoin(destination,'trial_info.csv'))
    return mmap_dat(dat_path), frames_avg, trialonsets,trialinfo


#######################################################################
################        For handling file sequences ###################
#######################################################################

class GenericStack():
    def __init__(self,filenames,extension):
        self.filenames = filenames
        self.fileextension = extension
        self.dims = None
        self.dtype = None
        self.frames_offset = []
        self.files = []
        self.current_fileidx = None
        self.current_stack = None

    def _get_frame_index(self,frame):
        '''
        Finds out in which file some frames are.
        '''
        fileidx = np.where(self.frames_offset <= frame)[0][-1]
        return fileidx,frame - self.frames_offset[fileidx]
    
    def _load_substack(fileidx):
        pass
    
    def _get_frame(self,frame):
        ''' 
        Returns a single frame from the stack.
        '''
        fileidx,frameidx = self._get_frame_index(frame)
        if not fileidx == self.current_fileidx:
            self._load_substack(fileidx)
        
        return self.current_stack[frameidx]

    def __len__(self):
        return self.shape[0
        ]
    def __getitem__(self, *args, squeeze = False):
        index  = args[0]
        idx1 = None
        idx2 = None
        if type(index) is tuple: # then look for 2 channels
            if type(index[1]) is slice:
                idx2 = range(index[1].start, index[1].stop, index[1].step)
            else:
                idx2 = index[1]
            index = index[0]
        if type(index) is slice:
            idx1 = range(*index.indices(self.nframes))#start, index.stop, index.step)
        elif type(index) in [int,np.int32, np.int64]: # just a frame
            idx1 = [index]
        else: # np.array?
            idx1 = index
        img = np.empty((len(idx1),*self.dims),dtype = self.dtype)
        for i,ind in enumerate(idx1):
            img[i] = self._get_frame(ind)
        if not idx2 is None:
            if squeeze:
                return img[:,idx2].squeeze()
            else:
                return img[:,idx2]
        if squeeze:
            return img.squeeze()
        else:
            return img

    def export_binary(self, foldername,
                      basename = 'frames',
                      chunksize = 512,
                      start_frame = 0,
                      end_frame = None,
                      channel = None):
        '''
        Exports a binary file.
        '''
        nframes,nchan,h,w = self.shape
        if end_frame is None :
            end_frame = nframes
        nframes = end_frame - start_frame
        chunks = chunk_indices(nframes,chunksize)    
        chunks = [[c[0]+start_frame,c[1]+start_frame] for c in chunks]
        shape = [nframes,*self.shape[1:]]
        if not channel is None:
            shape[1] = 1
        fname = pjoin('{0}'.format(foldername),'{4}_{0}_{1}_{2}_{3}.bin'.format(
            *shape[1:],self.dtype,basename))
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        out = np.memmap(fname,
                        dtype = self.dtype,
                        mode = 'w+',
                        shape=tuple(shape))
        for c in tqdm(chunks, desc='Exporting binary'):
            if channel is None:
                out[c[0] - start_frame:c[1] - start_frame] = self[c[0]:c[1]]
            else:
                out[c[0] - start_frame:c[1] - start_frame,0] = self[c[0]:c[1],channel]
        out.flush()
        del out

    def export_tiffs(self, foldername,
                     basename = 'frames',
                     chunksize = 512,
                     start_frame = 0,
                     end_frame = None,
                     channel = None):
        '''
        Exports tifffiles.
        '''
        nframes,nchan,h,w = self.shape
        if end_frame is None :
            end_frame = nframes
        nframes = end_frame - start_frame
        chunks = chunk_indices(nframes,chunksize)    
        chunks = [[c[0]+start_frame,c[1]+start_frame] for c in chunks]
        shape = [nframes,*self.shape[1:]]
        if not channel is None:
            shape[1] = 1

        file_no = 0
        fname = pjoin('{0}'.format(foldername),'{0}_{1:05d}.tif')
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        from tifffile import imsave
        for c in tqdm(chunks, desc='Exporting tiffs'):
            if channel is None:
                imsave(fname.format(basename,file_no),self[c[0]:c[1]].reshape((-1,*self.dims[1:])))
            else:
                imsave(fname.format(basename,file_no),self[c[0]:c[1],channel].squeeze())
            file_no += 1

        
class ImagerStack(GenericStack):
    def __init__(self,filenames,
                 extension = '.dat',
                 version = 2,# this is because the triggers number changed between the new and the old version...
                 rotate_array=True):
        '''
        
        rotate_array=True is for rotating the files saved by the imager...
        '''
        self.rotate_array = rotate_array
        self.fileformat = 'binary'
        self.version = version
        self.extension = extension
        if type(filenames) is str:
            # check if it is a folder
            if os.path.isdir(filenames):
                dirname = filenames
                filenames = natsorted(glob(pjoin(dirname,'Frames*'+self.extension)))
                if not len(filenames): # try mj2's
                    self.extension = '.mj2'
                    filenames = natsorted(glob(pjoin(dirname,'Frames*'+self.extension)))
                    if len(filenames):
                        self.fileformat = 'mj2'
                        self.rotate_array = False # This is not needed for these files...
                    else:
                        raise('Could not find files.')
        super(ImagerStack,self).__init__(filenames,extension)
        
        self.index_ch1 = []
        self.index_ch2 = []
        self.extrainfo = []
        for f in tqdm(self.filenames,desc='Parsing files to access the stack size'):
            # Parse all analog files and frames
            ch1,ch2,info = _imager_parse_file(f,version = self.version)
            self.index_ch1.append(ch1)
            self.index_ch2.append(ch2)
            self.extrainfo.append(info)
        # offset for each file
        self.frames_offset = np.hstack([0,np.cumsum([len(x) for x in self.index_ch1])])
        # get the dims from the first binary file
        if self.fileformat == 'mj2':
            stack = read_mj2_frames(f)
        else:
            stack = mmap_dat(f)
        if self.rotate_array: # fix imager rotation
            if len(stack.shape) == 3:
                stack = stack.transpose([0,2,1])
            if len(stack.shape) == 4:
                stack = stack.transpose([0,1,3,2])
        self.dims = stack.shape[1:]
        self.dtype = stack.dtype
        self.nframes = self.frames_offset[-1]
        self.shape = tuple([self.nframes,*self.dims])
        
    def _load_substack(self,fileidx,channel = None):
        if self.fileformat == 'mj2':
            tmp = read_mj2_frames(self.filenames[fileidx])
        else:
            tmp = load_dat(self.filenames[fileidx])
        if self.rotate_array: # fix imager rotation
            if len(tmp.shape) == 3:
                tmp = tmp.transpose([0,2,1])
            if len(tmp.shape) == 4:
                tmp = tmp.transpose([0,1,3,2])
        tmp = tmp.reshape([-1,*tmp.shape[-2:]])
        # combine the indexes from the 2 channels
        idx = np.sort(np.hstack([self.index_ch1[fileidx],
                                 self.index_ch2[fileidx]]))
        try:
            self.current_stack = tmp[idx].reshape([-1,*self.dims])
        except:
            raise OSError('There is a chance that file {0} is corrupt.'.format(
                self.filenames[fileidx]))
        self.current_fileidx = fileidx
                    
class BinaryStack(GenericStack):
    def __init__(self,filenames,
                 extension = '.bin'): # this will try this extension first and then .dat
                
        '''
        Select a stack from a binary file or mutliple binary files
        The file name format needs to ends in _NCHANNELS_H_W_DTYPE.extension

        '''
        self.fileformat = 'binary'
        self.extension = extension
        if type(filenames) is str:
            # check if it is a folder
            if os.path.isdir(filenames):
                dirname = filenames
                filenames = []
                filenames = natsorted(glob(pjoin(dirname,'*'+self.extension)))
                if not len(filenames): # try .dat
                    self.extension = '.dat'
                    filenames = natsorted(glob(pjoin(dirname,'*'+self.extension)))
        if not len(filenames):
            raise(OSError('Could not find files.'))
                
        super(BinaryStack,self).__init__(filenames,extension)
        offsets = [0]
        for f in tqdm(self.filenames, desc='Parsing files to know the stack size'):
            # Parse all binary files
            tmp =  mmap_dat(f)
            dims = tmp.shape[1:]
            dtype = tmp.dtype
            offsets.append(tmp.shape[0])
            del tmp
        # offset for each file
        self.frames_offset = np.cumsum(offsets)

        self.dims = dims
        self.dtype = dtype
        self.nframes = self.frames_offset[-1]
        self.shape = tuple([self.nframes,*self.dims])
        
    def _load_substack(self,fileidx,channel = None):
        self.current_stack = mmap_dat(self.filenames[fileidx])
        self.current_fileidx = fileidx

        
class TiffStack(GenericStack):
    def __init__(self,filenames,
                 extension = '.tiff', # this will try this extension first and then .tif, .TIFF and .TIF
                 nchannels = 2): 
        '''
        Select a stack from a sequence of TIFF stack files

        '''
        self.extension = extension
        if type(filenames) is str:
            # check if it is a folder
            if os.path.isdir(filenames):
                dirname = filenames
                filenames = []
                for extension in [self.extension,'.tif','.TIFF','.TIF']:
                    if not len(filenames): # try other
                        self.extension = extension
                        filenames = natsorted(glob(pjoin(dirname,'*'+self.extension)))
        if not len(filenames):
            raise(OSError('Could not find files.'))
        super(TiffStack,self).__init__(filenames,extension)
        from tifffile import imread, TiffFile
        self.imread = imread
        offsets = [0]
        for f in tqdm(self.filenames, desc='Parsing tiffs'):
            # Parse all files in the stack
            tmp =  TiffFile(f)
            dims = [*tmp.series[0].shape]
            if len(dims) == 2: # then these are single page tiffs
                dims = [1,*dims]
            dtype = tmp.series[0].dtype
            offsets.append(dims[0])
            del tmp
        # offset for each file
        self.frames_offset = np.cumsum(offsets)
        if nchannels is None:
            nchannels = 1
        self.frames_offset = (self.frames_offset/nchannels).astype(int)
        self.dims = dims[1:]
        if len(self.dims) == 2:
            self.dims = [nchannels,*self.dims]
        self.dims[0] = nchannels
        self.dtype = dtype
        self.nframes = self.frames_offset[-1]
        self.shape = tuple([self.nframes,*self.dims])
        
    def _load_substack(self,fileidx,channel = None):
        self.current_stack = self.imread(self.filenames[fileidx]).reshape([-1,*self.dims])
        self.current_fileidx = fileidx



        

def load_stack(foldername,order = ['binary','tiff','imager'], nchannels=None):
    ''' 
    Searches the correct format to load from a folder.
    '''
    # First check whats in the folder
    if os.path.isfile(foldername):
        if foldername.endswith('.bin') or  foldername.endswith('.dat'): 
            return mmap_dat(foldername)
    # Check binary sequence.
    files = natsorted(glob(pjoin(foldername,'*.bin')))
    if len(files):
        # these don't need channel number because it is written with the filename
        if len(files) == 1:
            return mmap_dat(files[0])
        print('Loading binary stack.')
        return BinaryStack(files) 
    # check tiff sequence
    for ext in ['.TIFF','.TIF','.tif','.tiff']:
        files = natsorted(glob(pjoin(foldername,'*'+ext)))
        if len(files):
            return TiffStack(files, nchannels = nchannels)
    # check for avi and mov
    
    # check imager
    files = natsorted(glob(pjoin(foldername,'Analog*.dat')))
    if len(files):
        return ImagerStack(foldername)
    # check for dat
    files = natsorted(glob(pjoin(foldername,'*.dat')))
    if len(files):
        if len(files) == 1:
            return mmap_dat(files[0])
        return BinaryStack(foldername)

