# This is the command line interface 
import argparse
import sys
import os
import subprocess
import shlex
import platform
from .utils import *
from .io import parse_imager_mj2_folder,mmap_dat
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
    open                Opens a gui to look at the preprocessed data        
    preprocess          Preprocess data in binary fornat
    imager              Preprocesses data recorded with the WidefieldImager
    motion              Registers data from a binary file
    decompose           Performs single value decomposition
    correct             Performs hemodynamic correction on dual channel data

''')
        parser.add_argument('command', help='type wfieldtools <command> -h for help')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Command {0} not recognized.'.format(args.command))
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def open(self):
        parser = argparse.ArgumentParser(
            description='Open the GUI to look at preprocessed data',
            usage = '''

            Press control while moving the mouse around to look at the timecourse at different points.

            Right click and drag mouse on the trace plot to zoom.
''')
        parser.add_argument('foldername', action='store',default=None,type=str)
        parser.add_argument('--before-corr', action='store_true',
                            default=False,
                            help= 'Load SVT before hemodynamics correction.')
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        from .io import mmap_dat
        fname = pjoin(localdisk,'Ua.npy')
        if os.path.isfile(fname):
            U = np.load(fname)
            if not args.before_corr:
                SVT = np.load(pjoin(localdisk,'SVTcorr.npy'))
            else:
                SVT = np.load(pjoin(localdisk,'SVTa.npy'))
        else:
            print('Could not find: {0} '.format(fname))
        dat_path = glob(pjoin(localdisk,'*.dat'))[0]
        dat = mmap_dat(dat_path)
        stack = SVDStack(U,SVT,dat.shape[-2:])
        del dat
        from .widgets import QApplication,SVDViewer
        app = QApplication(sys.argv)
        w = SVDViewer(stack)
        sys.exit(app.exec_())

    def imager(self):
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
        parser.add_argument('--mask-edge', action='store',default=30,type=int,
                            help = 'Size of the mask used on the edges during motion correction ') 
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
        _motion(localdisk,args.mask_edge)
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
        parser.add_argument('--mask-edge', action='store',default=30,type=int,
                            help = 'Size of the mask used on the edges during motion correction ') 
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
        _motion(localdisk,args.mask_edge)
        # COMPUTE AVERAGE FOR BASELINE
        _baseline(localdisk,args.nbaseline_frames)
        # DATA REDUCTION
        _decompose(localdisk,k=args.k)
        # HEMODYNAMIC CORRECTION
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
        parser.add_argument('--mask-edge', action='store',default=30,type=int,
                            help = 'Size of the mask used on the edges during motion correction ') 
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername 
        _motion(localdisk,args.mask_edge)
    def decompose(self):
        parser = argparse.ArgumentParser(
            description='Performs single value decomposition')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder with dat file.')
        parser.add_argument('-k', action='store',default=200,type=int,
                            help = 'Number of components for SVD ') 
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername 
        _decompose(localdisk,k=args.k)
    def correct(self):
        parser = argparse.ArgumentParser(
            description='Performs hemodynamics correction')
        parser.add_argument('foldername', action='store',
                            default=None, type=str,
                            help='Folder with U and SVT files.')
        parser.add_argument('--fs', action='store',default=30.,type=np.float32,
                            help = 'Sampling rate of an individual channel ') 
        
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername 

        _hemocorrect(localdisk,fs=args.fs)
        
def _motion(localdisk,mask_edge):
    dat_path = glob(pjoin(localdisk,'*.dat'))[0]        
    dat = mmap_dat(dat_path, mode='r+')
    yshifts,xshifts,avg_dat = motion_correct(dat,chunksize=512,
                                             mask_edge=mask_edge,
                                             apply_shifts=True)
    del dat # close and finish writing
    shifts = np.rec.array([yshifts,xshifts],dtype=[('y','int'),('x','int')])
    np.save(pjoin(localdisk,'motion_correction_shifts.npy'),shifts)
    np.save(pjoin(localdisk,'frames_average.npy'),avg_dat)
    del shifts

def _baseline(localdisk,nbaseline_frames):
    dat_path = glob(pjoin(localdisk,'*.dat'))[0]
    dat = mmap_dat(dat_path)
    try:
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))
    except FileNotFoundError:
        print('Skipping trial frame average because there was no trial_onsets.npy in the folder.')
        del dat
        return
    frames_average_trials = frames_average_for_trials(dat,
                                                      trial_onsets['iframe'],
                                                      nbaseline_frames)
    
    np.save(pjoin(localdisk,'frames_average.npy'),
            frames_average_trials.mean(axis=0))
    del dat
    del frames_average_trials

def _decompose(localdisk, k):
    dat_path = glob(pjoin(localdisk,'*.dat'))[0]
    frames_average = np.load(pjoin(localdisk,'frames_average.npy'))
    if len(frames_average)>3:
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))
        onsets = trial_onsets['iframe']

    else:
        onsets = None
    dat = mmap_dat(dat_path) # load to memory if you have enough
    U,SVT = approximate_svd(dat, frames_average,onsets = onsets,k = k)
    np.save(pjoin(localdisk,'Ua.npy'),U)
    np.save(pjoin(localdisk,'SVTa.npy'),SVT)

def _hemocorrect(localdisk,fs):
    U = np.load(pjoin(localdisk,'Ua.npy'))
    SVT = np.load(pjoin(localdisk,'SVTa.npy'))

    SVTcorr, rcoeffs, T = hemodynamic_correction(U, SVT, fs=fs)        

    np.save(pjoin(localdisk,'rcoeffs.npy'),rcoeffs)
    np.save(pjoin(localdisk,'T.npy'),T)
    np.save(pjoin(localdisk,'SVTcorr.npy'),SVTcorr)
            
    
def main():
    CLIParser()

if __name__ == '__main__':
    main()
