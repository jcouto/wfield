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
import argparse
import subprocess
import shlex
import platform
from .io import load_stack, mmap_dat
from .io import frames_average_for_trials
from .registration import motion_correct
from .decomposition import approximate_svd
from .hemocorrection import hemodynamic_correction

class CLIParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='wfield - tools to process widefield data',
            usage='''wfield <command> [<args>]

The commands are:
    ncaas               Opens a gui to launch data on the neuroCAAS platform
    open                Opens a gui to look at the preprocessed data        
    open_raw            Opens a gui to look at the raw frames
    preprocess          Preprocess data in binary fornat
    motion              Registers data from a binary file
    decompose           Performs single value decomposition
    correct             Performs hemodynamic correction on dual channel data
    imager              
    imager_preprocess   Preprocesses data recorded with the WidefieldImager
''')
        parser.add_argument('command', help='type wfieldtools <command> -h for help')

        args = parser.parse_args(sys.argv[1:2])
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
        trial_onsets = pjoin(localdisk,'trial_onsets.npy')
        if os.path.isfile(trial_onsets):
            trial_onsets = np.load(trial_onsets).astype(int)
        else:
            trial_onsets = None
        from .widgets import QApplication,RawViewer
        app = QApplication(sys.argv)
        w = RawViewer(raw = dat,
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
        dat,frames_avg, trialonsets,trialinfo = parse_imager_mj2_folder(remotepath, localdisk)
        del dat
        del frames_avg
        del trialonsets
        del trialinfo
        tfetch = (time.time() - tfetch)/60.

        tproc = time.time()
        # MOTION CORRECTION
        _motion(localdisk)
        # COMPUTE AVERAGE FOR BASELINE
        _baseline(localdisk,args.nbaseline_frames)
        # DATA REDUCTION
        _decompose(localdisk,k=args.k)
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
     
           Example name: frames_2_540_640_uint16.dat
    '''
)
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder that has the binary file (FAST DISK).')
        parser.add_argument('-k', action='store',default=200,type=int,
                            help = 'Number of components for SVD')
        #parser.add_argument('--mask-edge', action='store',default=30,type=int,
        #                    help = 'Size of the mask used on the edges during motion correction ') 
        parser.add_argument('--nchannels', action='store', default=None, type=int)
        parser.add_argument('--nbaseline-frames', action='store',
                            default=30, type=int,
                            help='Number of frames to compute the  baseline')
        parser.add_argument('--fs', action='store',default=30.,type=np.float32,
                            help='Sampling frequency of an individual channel')
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername # this should be an SSD or a fast drive

        if localdisk is None:
            print('Specify a fast local disk with the  -o option.')
            exit(1)
        if not os.path.isdir(localdisk):
            os.makedirs(localdisk)
            print('Created {0}'.format(localdisk))

        tproc = time.time()
        # MOTION CORRECTION
        _motion(localdisk)
        # COMPUTE AVERAGE FOR BASELINE
        _baseline(localdisk,args.nbaseline_frames)
        # DATA REDUCTION
        _decompose(localdisk,k=args.k)
        # HEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = load_stack(localdisk, nchannels = args.nchannels)
        if dat.shape[1] == 2:
            del dat
            _hemocorrect(localdisk,fs=args.fs)
            
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
        parser.add_argument('--mode', choices=('ecc','2d'),default='ecc',
                            help = 'Algorithm for  motion correction ') 
        parser.add_argument('--chunksize', default=256,
                            help = 'Frames per batch ') 
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername 
        _motion(localdisk,nchannels = args.nchannels, mode = args.mode,chunksize = args.chunksize)

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
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        if not args.no_baseline:
            _baseline(localdisk,args.nbaseline_frames, nchannels = args.nchannels)
        _decompose(localdisk,k=args.k, nchannels = args.nchannels)

    def correct(self):
        parser = argparse.ArgumentParser(
            description='Performs hemodynamics correction (needs 2 channel compressed data)')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder with U and SVT files.')
        parser.add_argument('--fs', action='store',default=30.,type=np.float32,
                            help = 'Sampling rate of an individual channel ') 
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername 

        _hemocorrect(localdisk,fs=args.fs)
        
def _motion(localdisk, nchannels = None, mode = 'ecc',chunksize=256, in_place = True):

    # if a single binary file try in-place
    dat_path = glob(pjoin(localdisk,'*.bin'))
    if not len(dat_path):
        dat_path = glob(pjoin(localdisk,'*.dat'))
    if len(dat_path) == 1:
        dat = mmap_dat(dat_path[0], mode='r+')
        if in_place:
            out = dat
    else:
        # else do else
        dat = load_stack(localdisk, nchannels = nchannels)
        out = np.memmap(pjoin(localdisk,'motioncorrect_{0}_{1}_{2}_{3}.bin'.format(*dat.shape[1:],dat.dtype)),
                        mode='w+',
                        dtype=dat.dtype,
                        shape = dat.shape)
    (yshifts,xshifts),rshifts = motion_correct(dat,
                                               out=out,
                                               chunksize=256,
                                               mode = mode,
                                               apply_shifts=True)
    del out # close and finish writing
    shifts = np.rec.array([yshifts,xshifts],dtype=[('y','float32'),('x','float32')])
    np.save(pjoin(localdisk,'motion_correction_shifts.npy'),shifts)
    np.save(pjoin(localdisk,'motion_correction_rotation.npy'),rshifts)
    from .plots import plot_summary_motion_correction
    plot_summary_motion_correction(shifts,localdisk)
    del shifts

def _baseline(localdisk, nbaseline_frames, nchannels = None):

    dat = load_stack(localdisk,nchannels = nchannels)
    try:
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))[:,1].astype(int)
    except FileNotFoundError:
        print('''
Skipping trial frame average because there was no trial_onsets.npy in the folder.
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

def _decompose(localdisk, k, nchannels = None):
    dat = load_stack(localdisk,nchannels = nchannels)

    frames_average = np.load(pjoin(localdisk,'frames_average.npy'))
    if len(frames_average)>3:
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))
        onsets = trial_onsets[:,1].astype(int)

    else:
        onsets = None
    U,SVT = approximate_svd(dat, frames_average, onsets = onsets, k = k)
    np.save(pjoin(localdisk,'U.npy'),U)
    np.save(pjoin(localdisk,'SVT.npy'),SVT)

def _hemocorrect(localdisk,fs):
    U = np.load(pjoin(localdisk,'U.npy'))
    SVT = np.load(pjoin(localdisk,'SVT.npy'))
    
    SVT_470 = SVT[:,0::2]
    t = np.arange(SVT.shape[1]) # interpolate the violet
    from scipy.interpolate import interp1d
    SVT_405 = interp1d(t[1::2],SVT[:,1::2],axis=1,
                       fill_value='extrapolate')(t[0::2])
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
                                          outputdir = localdisk)            
    
def main():
    CLIParser()

if __name__ == '__main__':
    main()
