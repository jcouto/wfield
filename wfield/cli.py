#  wfield - tools to analyse widefield data - command line interface 
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
from .io import load_stack, mmap_dat
from .io import frames_average_for_trials
from .registration import motion_correct
from .decomposition import approximate_svd
from .hemocorrection import hemodynamic_correction
import argparse
import subprocess
import shlex
import platform
from shutil import copy
from multiprocessing import set_start_method

class CLIParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='wfield - tools to process widefield data',
            usage='''wfield <command> [<args>]

The commands are:
    ncaas               Opens a gui to launch data on the neuroCAAS platform

    open                Opens a gui to look at the preprocessed data        
    open_raw            Opens a gui to look at the raw frames

    preprocess          Preprocess stack 

    motion              Registers stack
    decompose           Performs single value decomposition
    correct             Performs hemodynamic correction on dual channel data
    locanmf             Performs locaNMF decomposition on a pre-processed dataset

    imager              
    imager_preprocess   Preprocesses data recorded with the WidefieldImager
''')
        parser.add_argument('command', help='type wfieldtools <command> -h for help')

        args = parser.parse_args(sys.argv[1:2])
        set_start_method("spawn")
        if not hasattr(self, args.command):
            print('Command {0} not recognized.'.format(args.command))
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def ncaas(self):
        parser = argparse.ArgumentParser(
            description='Open the GUI to interact with neuroCAAS.org',
            usage = '''
You'll need credentials from neurocaas.org before you are able to use this.

Type wfield ncaas <foldername> to open on a specific folder.
            ''')
        parser.add_argument('--foldername', action='store',default='.',type=str)
        args = parser.parse_args(sys.argv[2:])
        
        from wfield.ncaas_gui import main as ncaas_gui

        ncaas_gui(args.foldername)
        
        
    def open(self):
        parser = argparse.ArgumentParser(
            description='Open the GUI to look at preprocessed data',
            usage = '''

            Press control while moving the mouse around to look at the timecourse at different points.

            Right click and drag mouse on the trace plot to zoom.
''')
        parser.add_argument('foldername', action='store',default=None,type=str)
        parser.add_argument('--allen-reference', action='store',default='dorsal_cortex',type=str)
        parser.add_argument('--no-ncaas', action='store_true',default=False)
        parser.add_argument('--correlation', action='store_true',default=False,help = 'Show the correlation window - dont do this with sparse matrices.')
        parser.add_argument('--nchannels', action='store', default=None, type=int)
        parser.add_argument('--before-corr', action='store_true',
                            default=False,
                            help= 'Load SVT before hemodynamics correction.')
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        from .io import mmap_dat
        fname = pjoin(localdisk,'U.npy')
        Uwarped = None
        if os.path.isdir(pjoin(localdisk,'results')) and not args.no_ncaas:
            # then it is from neurocaas?
            fname = pjoin(localdisk,'results','U_atlas.npy')
            if os.path.isfile(fname):
                Uwarped = np.load(fname)
                print('Found U_atlas')
                iswarped = True
            fname = pjoin(localdisk,'results','config.yaml')
            if os.path.isfile(fname):
                with open(fname,'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                H,W = (config['fov_height'],config['fov_width'])
                fname = pjoin(localdisk,'results','reduced_spatial.npy')
                if os.path.isfile(fname):
                    U = np.load(fname)
                    U = U.reshape([H,W,-1])
                    dims = [H,W]
            fname = pjoin(localdisk,'results','SVTcorr.npy')
            if not os.path.isfile(fname):
                fname = pjoin(localdisk,'results','reduced_temporal.npy')
            SVT = np.load(fname)
        elif os.path.isfile(fname):
            U = np.load(fname)
            dims = U.shape[:2]
            if (not args.before_corr
                and os.path.isfile(pjoin(localdisk,'SVTcorr.npy'))):
                SVT = np.load(pjoin(localdisk,'SVTcorr.npy'))
            else:
                SVT = np.load(pjoin(localdisk,'SVT.npy'))
        else:
            # try in a results folder (did it come from ncaas?)
            raise OSError('Could not find: {0} '.format(fname))


        dat = load_stack(args.foldername, nchannels = args.nchannels)
        #dat_path = glob(pjoin(localdisk,'*.bin'))
        #if len(dat_path):
        #    dat_path = dat_path[0]
        #    if os.path.isfile(dat_path):
        #        dat = mmap_dat(dat_path)
        if dat is None:
            dat = None
            dat_path = None
            average_path = pjoin(localdisk,'frames_average.npy')
            if os.path.isfile(average_path):
                dat = np.load(average_path)
                dat = dat.reshape([1,*dat.shape])

        maskpath = pjoin(localdisk,'manual_mask.npy')
        if os.path.isfile(maskpath):
            mask = np.load(maskpath).astype(int)
        else:
            mask = None

        trial_onsets = pjoin(localdisk,'trial_onsets.npy')
        if os.path.isfile(trial_onsets):
            trial_onsets = np.load(trial_onsets).astype(int)
        else:
            trial_onsets = None
        
        stack = SVDStack(U,SVT, warped = Uwarped, dims = dims)
        if not Uwarped is None:
            stack.set_warped(True)
        from .widgets import QApplication,SVDViewer
        app = QApplication(sys.argv)
        w = SVDViewer(stack,
                      folder = localdisk,
                      raw = dat,
                      mask = mask,
                      trial_onsets = trial_onsets,
                      reference = args.allen_reference,
                      start_correlation = args.correlation)
        sys.exit(app.exec_())
        del dat
        
    def open_raw(self):
        parser = argparse.ArgumentParser(
            description='Inspect the raw video frames')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder where to search for files.')
        parser.add_argument('--nchannels', action='store', default=None, type=int)
        parser.add_argument('--allen-reference', action='store',default='dorsal_cortex',type=str)
        parser.add_argument('--napari', action='store_true',
                            default=False,
                            help='Show with napari')
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        dat = load_stack(args.foldername, nchannels = args.nchannels)

        if os.path.isdir(localdisk):
            dat_path = glob(pjoin(localdisk,'*.bin'))
            if len(dat_path):
                dat_path = dat_path[0]
        else:
            dat_path = localdisk
            localdisk = os.path.dirname(localdisk)

        if args.napari:
            from .viz import napari_show
            napari_show(dat)
            del dat
            sys.exit()

        if dat is None:
            average_path = pjoin(localdisk,'frames_average.npy')
            if os.path.isfile(average_path):
                dat = np.load(average_path)
            dat = dat.reshape([1,*dat.shape])
            print('Loading the frames_average instead')
            average_path = pjoin(localdisk,'frames_average.npy')
            if os.path.isfile(average_path):
                dat = np.load(average_path)
            dat = dat.reshape([1,*dat.shape])
            print('Loading the frames_average instead')

        maskpath = pjoin(localdisk,'manual_mask.npy')
        if os.path.isfile(maskpath):
            mask = np.load(maskpath).astype(int)
        else:
            mask = None

        trial_onsets = pjoin(localdisk,'trial_onsets.npy')
        if os.path.isfile(trial_onsets):
            trial_onsets = np.load(trial_onsets).astype(int)
        else:
            trial_onsets = None

        
        from .widgets import QApplication,RawViewer
        app = QApplication(sys.argv)
        w = RawViewer(raw = dat,
                      mask = mask,
                      folder = localdisk,
                      trial_onsets = trial_onsets,
                      reference = args.allen_reference)
        sys.exit(app.exec_())
        del dat

        
    def imager(self):
        parser = argparse.ArgumentParser(
            description='Converts widefield data recorded with the WidefieldImager')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder where to search for mj2 and analog files.')
        parser.add_argument('-o','--output', action='store',
                            default=None, type=str,
                            help='Output folder') # there should be an intermediate folder as well
        
        args = parser.parse_args(sys.argv[2:])
        remotepath = args.foldername
        localdisk = args.output # this should be an SSD or a fast drive

        if localdisk is None:
            print('Specify a fast local disk with the  -o option.')
            exit(1)
        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print('Created {0}'.format(localdisk))
        print('''

    Searching for mj2 files in: 
        {0}
        
    Output folder: 
        {1}

        '''.format(remotepath,localdisk))
        tfetch = time.time()
        from .io import parse_imager_mj2_folder
        dat,frames_avg, trialonsets,trialinfo = parse_imager_mj2_folder(remotepath, localdisk)
        del dat
        del frames_avg
        del trialonsets
        del trialinfo
        tfetch = (time.time() - tfetch)/60.

        
    def imager_preprocess(self):
        parser = argparse.ArgumentParser(
        description='Performs preprocessing of widefield data recorded with the WidefieldImager')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder where to search for mj2 and analog files.')
        parser.add_argument('-o','--output', action='store',
                            default=None, type=str,
                            help='Output folder') # there should be an intermediate folder as well
        parser.add_argument('-k', action='store',default=200,type=int,
                            help = 'Number of components for SVD')
        #parser.add_argument('--mask-edge', action='store',default=30,type=int,
        #                    help = 'Size of the mask used on the edges during motion correction ') 
        parser.add_argument('--nbaseline-frames', action='store',
                            default=30, type=int,
                            help='Number of frames to compute the  baseline')
        parser.add_argument('--fs', action='store',default=30.,type=np.float32,
                            help='Sampling frequency of an individual channel')
        parser.add_argument('--mode', choices=('ecc','2d'),default='2d',
                            help = 'Algorithm for  motion correction ')
        parser.add_argument('--chunksize', default=256,
                            help = 'Frames per batch ') 
        parser.add_argument('--std-mask-threshold', action='store',
                            default=0, type=float,
                            help='Percentile threshold for the std mask applied before decomposing U.')        
        parser.add_argument('--match-session', action='store',
                            default=None, type=str,
                            help='Folder with wfield results file that is used to match.')


        args = parser.parse_args(sys.argv[2:])
        remotepath = args.foldername
        localdisk = args.output # this should be an SSD or a fast drive

        if localdisk is None:
            print('Specify a fast local disk with the  -o option.')
            exit(1)

        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print('Created {0}'.format(localdisk))
        print('''

    Searching for mj2 files in: 
        {0}
        
    Output folder: 
        {1}

        '''.format(remotepath,localdisk))
        tfetch = time.time()
        from .io import parse_imager_mj2_folder
        dat,frames_avg, trialonsets,trialinfo = parse_imager_mj2_folder(remotepath, localdisk)
        del dat
        del frames_avg
        del trialonsets
        del trialinfo
        tfetch = (time.time() - tfetch)/60.

        tproc = time.time()
        # MOTION CORRECTION
        _motion(localdisk,
                outdisk = localdisk,
                chunksize = args.chunksize,
                mode = args.mode)
        # COMPUTE AVERAGE FOR BASELINE
        _baseline(localdisk,args.nbaseline_frames)
        # DATA REDUCTION
        _decompose(localdisk, k = args.k,
                   std_mask_threshold = args.std_mask_threshold,
                   match_session = args.match_session)
        # HEMODYNAMIC CORRECTION
        _hemocorrect(localdisk,fs=args.fs)
        tproc = (time.time() - tproc)/60.
        print('Done fetching data ({0} min) and pre-processing ({1} min)'.format(tfetch,tproc))
        exit(0)

    def preprocess(self):

        parser = argparse.ArgumentParser(
            description='Performs preprocessing of widefield data recorded from a binary file',
            usage = '''
            wfield preprocess FOLDERNAME

     The folder must contain a binary file; the end of the filename must be _NCHANNELS_H_W_DTYPE.dat
     
           Example name: frames_2_540_640_uint16.dat''')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder that has the binary file (FAST DISK).')
        parser.add_argument('-o','--output', action='store',
                            default='wfield_results', type=str,
                            help='Output folder')
        parser.add_argument('-k', action='store',default=200,type=int,
                            help = 'Number of components for SVD')
        #parser.add_argument('--mask-edge', action='store',default=30,type=int,
        #                    help = 'Size of the mask used on the edges during motion correction ') 
        parser.add_argument('--nchannels', action='store', default=None, type=int,
                            help = 'Number of recorded channels (interleaved)')
        parser.add_argument('--functional-channel', action='store',
                            default=0, type=int,
                            help='Index of the functional channel')
        parser.add_argument('--nbaseline-frames', action='store',
                            default=30, type=int,
                            help='Number of frames to compute the  baseline')
        parser.add_argument('--fs', action='store',default=30.,type=np.float32,
                            help='Sampling frequency of an individual channel')
        parser.add_argument('--mode', choices=('ecc','2d'),default='2d',
                            help = 'Algorithm for  motion correction ') 
        parser.add_argument('--chunksize', default=256,
                            help = 'Frames per batch ') 
        parser.add_argument('--std-mask-threshold', action='store',
                            default=0, type=float,
                            help='Percentile threshold for the std mask applied before decomposing U.')        
        parser.add_argument('--match-session', action='store',
                            default=None, type=str,
                            help='Folder with wfield results file that is used to match.')

        
        args = parser.parse_args(sys.argv[2:])
        datadisk = os.path.abspath(args.foldername) # this should be an SSD or a fast drive
        if os.path.isfile(datadisk):
            datafolder = os.path.dirname(datadisk)
        else:
            datafolder = datadisk
        
        localdisk = args.output # this should be an SSD or a fast drive
        if localdisk is None:
            print('Specify a fast local disk with the  -o option.')
            exit(1)
        if localdisk == 'wfield_results':
            localdisk = pjoin(datafolder,localdisk)
        else:
            localdisk = os.path.abspath(localdisk)
        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print('Created {0}'.format(localdisk))

        lmarks = glob(pjoin(datafolder,'*landmarks*.json'))
        if len(lmarks):
            print('Found a landmarks file, copying to the results folder..')
            copy(lmarks[0],localdisk)
        maskfile = glob(pjoin(datafolder,'manual_mask.npy'))
        if len(maskfile):
            print('Found a mask file, copying to the results folder..')
            copy(maskfile[0],localdisk)

        tproc = time.time()
        # MOTION CORRECTION
        _motion(datadisk,outdisk = localdisk,
                mode = args.mode,
                chunksize = args.chunksize,
                nchannels = args.nchannels)
        # COMPUTE AVERAGE FOR BASELINE
        _baseline(localdisk,args.nbaseline_frames, nchannels = args.nchannels)
        # DATA REDUCTION
        _decompose(localdisk, k = args.k,
                   nchannels = args.nchannels,
                   std_mask_threshold = args.std_mask_threshold,
                   match_session = args.match_session)
        # HEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = load_stack(localdisk, nchannels = args.nchannels)
        if dat.shape[1] == 2:
            del dat
            _hemocorrect(localdisk,fs=args.fs, functional_channel = args.functional_channel)
            
        tproc = (time.time() - tproc)/60.
        print('Done  pre-processing ({0} min)'.format(tproc))
        exit(0)
        
    def motion(self):
        parser = argparse.ArgumentParser(
            description='Performs motion correction on widefield data')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder with dat file.')
        parser.add_argument('--nchannels', action='store', default=None, type=int)
        parser.add_argument('--mode', choices=('ecc','2d'),default='2d',
                            help = 'Algorithm for  motion correction ') 
        parser.add_argument('--chunksize', default=256,
                            help = 'Frames per batch ') 

        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        _motion(localdisk,
                nchannels = args.nchannels,
                mode = args.mode,
                chunksize = args.chunksize)

    def decompose(self):
        parser = argparse.ArgumentParser(
            description='Performs single value decomposition')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder with dat file.')
        parser.add_argument('-k', action='store',default=200,type=int,
                            help = 'Number of components for SVD ') 
        parser.add_argument('--nchannels', action='store', default=None, type=int)
        parser.add_argument('--no-baseline',
                            action='store_true', default=False,
                            help = 'Skip baseline ') 
        parser.add_argument('--nbaseline-frames', action='store',
                            default=30, type=int,
                            help='Number of frames to compute the  baseline')        
        parser.add_argument('--std-mask-threshold', action='store',
                            default=0, type=float,
                            help='Percentile threshold for the std mask applied before decomposing U.')
        parser.add_argument('--match-session', action='store',
                            default=None, type=str,
                            help='Folder with wfield results file that is used to match.')

        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        if not args.no_baseline:
            _baseline(localdisk,args.nbaseline_frames, nchannels = args.nchannels)
        _decompose(localdisk,
                   k=args.k,
                   nchannels = args.nchannels,
                   std_mask_threshold = args.std_mask_threshold,
                   match_session = args.match_session)

    def correct(self):
        parser = argparse.ArgumentParser(
            description='Performs hemodynamics correction (needs 2 channel compressed data)')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder with U and SVT files.')
        parser.add_argument('--fs', action='store',default=30.,type=np.float32,
                            help = 'Sampling rate of an individual channel ') 
        parser.add_argument('--functional-channel', action='store',
                            default=0, type=int,
                            help='Index of the functional channel')
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername 

        _hemocorrect(localdisk, fs=args.fs,
                     functional_channel=args.functional_channel)
        
def _motion(localdisk,
            nchannels = None,
            mode = 'ecc',
            chunksize=256,
            outdisk = None,
            in_place = False,
            plot_ext = '.pdf'):
    # TODO: check if a motion corrected file is already here
    # if a single binary file try in-place
    dat_path = glob(pjoin(localdisk,'*.bin'))
    if outdisk is None:
        outdisk = localdisk
    if not len(dat_path):
        dat_path = glob(pjoin(localdisk,'*.dat'))
    if len(dat_path) == 1:
        dat = mmap_dat(dat_path[0], mode='r+')
    else:
        # else do else
        dat = load_stack(localdisk, nchannels = nchannels)
    if in_place:
        out = dat
    else:
        out = np.memmap(pjoin(outdisk,'motioncorrect_{0}_{1}_{2}_{3}.bin'.format(*dat.shape[1:],dat.dtype)),
                        mode='w+',
                        dtype=dat.dtype,
                        shape = dat.shape)
    (yshifts,xshifts),rshifts = motion_correct(dat,
                                               out=out,
                                               chunksize=int(chunksize),
                                               mode = mode,
                                               apply_shifts=True)
    del out # close and finish writing
    shifts = np.rec.array([yshifts,xshifts],dtype=[('y','float32'),('x','float32')])
    np.save(pjoin(outdisk,'motion_correction_shifts.npy'),shifts)
    np.save(pjoin(outdisk,'motion_correction_rotation.npy'),rshifts)
    from .plots import plot_summary_motion_correction
    plot_summary_motion_correction(shifts,outdisk, plot_ext=plot_ext)
    del shifts

def _baseline(localdisk, nbaseline_frames, nchannels = None):

    if os.path.exists(pjoin(localdisk,'frames_average.npy')):
        print('Found frame_average.npy. skipping.')
        
        return np.load(pjoin(localdisk,'frames_average.npy'))
    dat = load_stack(localdisk,nchannels = nchannels)
    try:
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))[:,1].astype(int)
    except FileNotFoundError:
        print('''
 Estimating the mean by the average of the chunked mean projection.
''')
        chunkidx = chunk_indices(len(dat),chunksize=64)
        frame_averages = []
        for on,off in tqdm(chunkidx):
            frame_averages.append(dat[on:off].mean(axis=0))
        del dat
        frames_average = np.stack(frame_averages).mean(axis = 0)
        np.save(pjoin(localdisk,'frames_average.npy'),
                frames_average)
        return frames_average
    frames_average_trials = frames_average_for_trials(dat,
                                                      trial_onsets,
                                                      nbaseline_frames)
    
    np.save(pjoin(localdisk,'frames_average.npy'),
            frames_average_trials.mean(axis=0))
    del dat
    del frames_average_trials

def _decompose(localdisk, k, nchannels = None,
               std_mask_threshold = 0,
               functional_channel=0,
               mask_from_atlas = True,
               match_session = None):
    dat = load_stack(localdisk,nchannels = nchannels)

    frames_average = np.load(pjoin(localdisk,'frames_average.npy'))
    if not match_session is None:
        print('Getting the transformation and mask from another session: ' + match_session)
        if not os.path.exists(match_session):
            raise(OSError(match_session + ' not found.'))
        from .multisession import prepare_multisession_match_files
        transform, nlmarks, nmask = prepare_multisession_match_files(localdisk, match_session)
        std_mask_threshold = 0
        mask_from_atlas = False
        
    if len(frames_average)>3:
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))
        onsets = trial_onsets[:,1].astype(int)

    else:
        onsets = None
        
    if os.path.exists(pjoin(localdisk,'mask.npy')):
        mask  = np.load(pjoin(localdisk,'mask.npy'))
    else:
        mask = np.zeros(dat.shape[-2::],dtype=bool)
    if mask_from_atlas: # get the mask from the landmarks file
        lmarks = glob(pjoin(localdisk,'*landmarks*.json'))
        if len(lmarks):
            from .allen import atlas_from_landmarks_file
            _, _, mask = atlas_from_landmarks_file(lmarks[0],dims = mask.shape, do_transform=True)
            print('Using the mask from the landmarks file for decomposition.')
    if std_mask_threshold > 0: # then compute the stdmask
        print("Using the standard deviation of the functional channel to compute the mask.")
        mask  = mask & get_std_mask(dat[:,functional_channel],threshold=std_mask_threshold)
    if os.path.exists(pjoin(localdisk,'manual_mask.npy')):
        manual = np.load(pjoin(localdisk,'manual_mask.npy')) # the manual mask is to remove parts of the image
        mask = (mask) & (manual==0)  # the manual mask is added to the std_mask_threshold
        print('Loading the manual mask')
    if np.sum(mask) == 0:
        mask = None
    else:
        np.save(pjoin(localdisk,'mask.npy'),mask)
        print('Saved mask file.')
    U,SVT = approximate_svd(dat, frames_average, onsets = onsets, mask = mask, k = k)
    np.save(pjoin(localdisk,'U.npy'),U)
    np.save(pjoin(localdisk,'SVT.npy'),SVT)

def _hemocorrect(localdisk,fs,functional_channel = 0, plot_ext='.pdf'):
    U = np.load(pjoin(localdisk,'U.npy'))
    SVT = np.load(pjoin(localdisk,'SVT.npy'))
    
    SVT_470 = SVT[:,np.mod(functional_channel,2)::2]
    t = np.arange(SVT.shape[1]) # interpolate the violet
    from scipy.interpolate import interp1d
    SVT_405 = interp1d(t[np.mod(functional_channel+1,2)::2],
                       SVT[:,np.mod(functional_channel+1,2)::2],axis=1,
                       fill_value='extrapolate')(
                           t[np.mod(functional_channel,2)::2])
    SVTcorr, rcoeffs, T = hemodynamic_correction(U, SVT_470, SVT_405, fs=fs)  

    np.save(pjoin(localdisk,'rcoeffs.npy'),rcoeffs)
    np.save(pjoin(localdisk,'T.npy'),T)
    np.save(pjoin(localdisk,'SVTcorr.npy'),SVTcorr)
    from .plots import plot_summary_hemodynamics_dual_colors
    plot_summary_hemodynamics_dual_colors(rcoeffs,
                                          SVT_470,
                                          SVT_405,
                                          U,
                                          T,
                                          frame_rate=fs,
                                          duration = 12,
                                          outputdir = localdisk,
                                          plot_ext = plot_ext)
    
def main():
    CLIParser()

if __name__ == '__main__':
    main()
