# These are the command line interface tools

import argparse
import sys
import os
import subprocess
import shlex
import platform
from .utils import *

hostname = platform.node()

class CLIParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='wfield - tools to process widefield data',
            usage='''wfield <command> [<args>]

The commands are:
    open                Opens a gui to look at the preprocessed data        
    imager              Preprocesses data recorded with the WidefieldImager
    register            Registers data from a binary file
    decompose           Performs single value decomposition
    correct             Performs hemodynamic correction on dual channel data
''')
        parser.add_argument('command', help='Subcommand to run')
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
            description='Open the GUI to look at preprocessed data')
        parser.add_argument('foldername', action='store',default=None,type=str)
        args = parser.parse_args(sys.argv[2:])
        localdisk = args.foldername
        from .io import mmap_dat
        fname = pjoin(localdisk,'Ua.npy')
        if os.path.isfile(fname):
            U = np.load(fname)
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
            description='Open the GUI to look at preprocessed data')
        parser.add_argument('foldername', action='store',default=None,type=str)
        parser.add_argument('-o','--output', action='store',default=None,type=str) # there should be an intermediate folder as well
        parser.add_argument('-k', action='store',default=200,type=int) # number of components
        parser.add_argument('--mask-edge', action='store',default=30,type=int) # mask edge for motion correction
        parser.add_argument('--nbaseline-frames', action='store',default=30,type=int) # number of baseline frames
        parser.add_argument('--fs', action='store',default=30.,type=np.float32) # sampling rate of a single channel

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
        from .io import parse_imager_mj2_folder,mmap_dat
        tfetch = time.time()
        dat,frames_avg, trialonsets,trialinfo = parse_imager_mj2_folder(remotepath, localdisk)
        del dat
        del frames_avg
        del trialonsets
        del trialinfo
        tfetch = (time.time() - tfetch)/60.

        tproc = time.time()

        # MOTION CORRECTION
        dat_path = glob(pjoin(localdisk,'*.dat'))[0]        
        dat = mmap_dat(dat_path, mode='r+')
        from .registration import motion_correct
        yshifts,xshifts,avg_dat = motion_correct(dat,chunksize=512,
                                                 mask_edge=args.mask_edge,
                                                 apply_shifts=True)
        del dat # close and finish writing
        shifts = np.rec.array([yshifts,xshifts],dtype=[('y','int'),('x','int')])
        np.save(pjoin(localdisk,'motion_correction_shifts.npy'),shifts)
        np.save(pjoin(localdisk,'frames_average.npy'),avg_dat)
        del shifts
        
        dat_path = glob(pjoin(localdisk,'*.dat'))[0]
        # COMPUTE AVERAGE FOR BASELINE
        dat = mmap_dat(dat_path)
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))
        from .io import frames_average_for_trials
        frames_average = frames_average_for_trials(dat,
                                                   trial_onsets['iframe'],
                                                   args.nbaseline_frames)
        np.save(pjoin(localdisk,'frames_average.npy'),frames_average)
        del dat
        del frames_average
        # DATA REDUCTION
        dat_path = glob(pjoin(localdisk,'*.dat'))[0]
        frames_average = np.load(pjoin(localdisk,'frames_average.npy'))
        trial_onsets = np.load(pjoin(localdisk,'trial_onsets.npy'))
        
        dat = mmap_dat(dat_path) # load to memory if you have enough
        onsets = trial_onsets['iframe']
        from .decomposition import approximate_svd
        U,SVT = approximate_svd(dat, frames_average,onsets = onsets)
        np.save(pjoin(localdisk,'Ua.npy'),U)
        np.save(pjoin(localdisk,'SVTa.npy'),SVT)
        from .hemocorrection import hemodynamic_correction
        SVTcorr, rcoeffs, T = hemodynamic_correction(U, SVT, fs=args.fs)        

        np.save(pjoin(localdisk,'rcoeffs.npy'),rcoeffs)
        np.save(pjoin(localdisk,'T.npy'),T)
        np.save(pjoin(localdisk,'SVTcorr.npy'),SVTcorr)
        del dat
        tproc = (time.time() - tproc)/60.
        print('Done fetching data ({0} min) and pre-processing ({1} min)'.format(tfetch,tproc))
        exit(0)

def main():
    CLIParser()

if __name__ == '__main__':
    main()
