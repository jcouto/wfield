#  wfield - tools to analyse widefield data - NCAAS gui
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
# 

from .utils import *
import sys
from glob import glob
import json
from datetime import datetime
import time
import threading
from PyQt5.QtWidgets import (QApplication,
                             QWidget,
                             QMainWindow,
                             QDockWidget,
                             QFormLayout,
                             QHBoxLayout,
                             QGridLayout,
                             QVBoxLayout,
                             QPushButton,
                             QGridLayout,
                             QTreeWidgetItem,
                             QTreeView,
                             QTextEdit,
                             QPlainTextEdit,
                             QLineEdit,
                             QScrollArea,
                             QCheckBox,
                             QComboBox,
                             QListWidget,
                             QLabel,
                             QProgressBar,
                             QFileDialog,
                             QMessageBox,
                             QDesktopWidget,
                             QListWidgetItem,
                             QFileSystemModel,
                             QAbstractItemView,
                             QTabWidget,
                             QMenu,
                             QDialog,
                             QDialogButtonBox,
                             QAction)
from functools import partial

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except:
    print('Could not load QWebEngineView.')
from PyQt5 import QtCore
from PyQt5.QtGui import QStandardItem, QStandardItemModel,QColor
from PyQt5.QtCore import Qt, QTimer,QMimeData

import yaml

try:
    import boto3
except:
    print('boto3 not installed, installing with pip ')
    from subprocess import call
    call('pip install boto3',shell = True)
    import boto3


tempfile = pjoin(os.path.expanduser('~'),'.wfield','tempfile')

analysis_extensions = ['.bin','.tiff','.tif','.npy','.json','.dat']

analysis_selection_dict = [dict(acronym = 'motion',
                                name='Motion correction',
                                desc = 'Performs the registration of frames to a reference taken from aligning the second 60 frames of the dataset to themselves.',
                                selected = True),
                           dict(acronym='compression',
                                name='Compression and denoising using PMD',
                                desc = 'Performs Penalized Matrix Decomposition (PMD) to compress and denoise the dataset. This step operates on motion corrected data.',
                                selected=True),
                           dict(acronym='hemodynamics_compensation',
                                name='Hemodynamics compensation',
                                desc = 'Performs hemodynamics compensation by regressing the second channel (isobestic for the indicator) into the first (indicator fluorescence) and subtracting the result to the first channel. This step requires a compressed dataset with 2 channels.',
                                selected = True),
                           dict(acronym='locaNMF',
                                name='Feature extraction using LocaNMF to isolate activity from individual sources in the data.',
                                desc = 'Applies locaNMF to the dataset to isolate the activity of individual areas. This step requires a compressed  dataset.',
                                selected = True)]


defaultconfig = {
    'cshl-wfield-preprocessing': dict(
        submit = dict(
            instance_type =  'r5.16xlarge',
            analysis_extension = '.bin',
            userfolder = 'ChurchlandLab',
            decompress_results = True),  # this will decompress the U matrix when downloading
        config = dict(block_height = 90,
                      block_width = 80,
                      frame_rate = 60,
                      max_components = 15,
                      num_sims = 64,
                      overlapping = True,
                      window_length = 200)), # 7200    
    'cshl-wfield-locanmf': {
        'submit':dict(
            instance_type =  'p3.2xlarge',
            userfolder = 'data',
            params_filename = 'config.yaml',
            areanames_filename = 'labels.json',
            atlas_filename = 'atlas.npy',
            brainmask_filename = 'brainmask.npy',
            temporal_data_filename = 'SVTcorr.npy',
            spatial_data_filename = 'U.npy'),
        'config' :{"maxrank": 3,
                   "loc_thresh": 80,
                   "min_pixels": 100,
                   "r2_thresh": 0.99,
	           "maxiter_hals":20,
	           "maxiter_lambda":300,
	           "lambda_step":1.35,
	           "lambda_init":0.000001}}}

# Function to add credentials
awsregions = ['us-east-2',
              'us-east-1',
              'us-west-1',
              'us-west-2',
              'af-south-1',
              'ap-east-1',
              'ap-south-1',
              'ap-northeast-3',
              'ap-northeast-2',
              'ap-southeast-1',
              'ap-southeast-2',
              'ap-northeast-1',
              'ca-central-1',
              'cn-north-1',
              'cn-northwest-1',
              'eu-central-1',
              'eu-west-1',
              'eu-west-2',
              'eu-south-1',
              'eu-west-3',
              'eu-north-1',
              'me-south-1',
              'sa-east-1']

def ncaas_set_aws_keys(access_key,secret_key,region='us-east-1'):
    fname = pjoin(os.path.expanduser('~'),'.aws','credentials')
    cred = '''[default]
aws_access_key_id = {access_key}
aws_secret_access_key = {secret_key}
'''
    dirname = os.path.dirname(fname)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)    
    with open(fname,'w') as fd:
        fd.write(cred.format(access_key=access_key,secret_key = secret_key))
    fname = pjoin(os.path.expanduser('~'),'.aws','config')
    if not region is None:
        conf = '''[default]
region={region}'''
        with open(fname,'w') as fd:
            fd.write(conf.format(region=region))

def ncaas_read_aws_keys():
    awsfolder = pjoin(os.path.expanduser('~'),'.aws')
    awscredfile = pjoin(awsfolder,'credentials')
    awsconfig = pjoin(awsfolder,'credentials')
    access_key = ''
    secret_key = ''
    region = 'us-east-1'
    if os.path.isfile(awscredfile):
        with open(awscredfile,'r') as fd:
            for ln in fd:
                if 'aws_access_key_id' in ln:
                    ln = ln.split('=')
                    if len(ln)>1:
                        access_key = ln[-1].strip(' ').strip('\n')
                if 'aws_secret_access_key' in ln:
                    ln = ln.split('=')
                    if len(ln)>1:
                        secret_key = ln[-1].strip(' ').strip('\n')
    if os.path.isfile(awsconfig):
        with open(awsconfig,'r') as fd:
            for ln in fd:
                if 'region' in ln:
                    ln = ln.split('=')
                    region = ln[-1].strip(' ')
    return dict(access_key = access_key,
                secret_key = secret_key,
                region = region)    

def ncaas_read_analysis_config(config):
    if not os.path.exists(os.path.dirname(config)):
        os.makedirs(os.path.dirname(config))
    if not os.path.exists(config):
        with open(config,'w') as f:
            print('Creating config from defaults [{0}]'.format(config))
            json.dump(defaultconfig,f,
                      indent = 4,
                      sort_keys = True)
    with open(config,'r') as f:
        config = json.load(f)
        for k in defaultconfig.keys():
            if not k in config.keys(): # Use default
                config[k] = defaultconfig[k]
    return config

def s3_connect():
    return boto3.resource('s3')

def s3_ls(s3,bucketnames):
    files = []
    for bucketname in bucketnames:
        bucket = s3.Bucket(bucketname)
        files.extend([bucketname+'/'+l.key for l in list(bucket.objects.all())])
    return files
    
def make_tree(item, tree):
    if len(item) == 1:
        if not item[0] == '':
            tree[item[0]] = item[0]
    else:
        head, tail = item[0], item[1:]
        tree.setdefault(head, {})
        make_tree(
            tail,
            tree[head])
def build_tree(item,parent):
    for k in item.keys():
        child = QStandardItem(k)
        child.setFlags(child.flags() |
                       Qt.ItemIsSelectable |
                       Qt.ItemIsEnabled)
        child.setEditable(False)
        if type(item[k]) is dict:
            build_tree(item[k],child)
        parent.appendRow(child)

def to_log(msg,logfile = None):
    if logfile is None:
        logfile = open(pjoin(
                     os.path.expanduser('~'),
                     '.wfield','ncaas_gui_log.txt'),'a')
    nmsg = '['+datetime.today().strftime('%y-%m-%d %H:%M:%S')+'] - ' + msg + '\n'
    logfile.seek(os.SEEK_END)
    logfile.write(nmsg)
    return nmsg

def tail(filename, nlines=100):
    """
    This needs work (should not read the whole file).
    """
    with open(filename,'r') as f:
        lines = f.readlines()
    if len(lines) > 100:
        lines = lines[-100:]
    return lines
    
class CredentialsManager(QDockWidget):
    def __init__(self,
                 configfile = pjoin(
                     os.path.expanduser('~'),
                     '.wfield','ncaas_config.json')):
        '''
        Logs to NeuroCaas to retrieve the keys and 
allows changing the config parameters and credentials 
from the GUI.
        '''
        super(CredentialsManager,self).__init__()
        self.awsinfo = ncaas_read_aws_keys()
        self.ncaasconfig = ncaas_read_analysis_config(configfile)
        ncaasconfig_json = json.dumps(self.ncaasconfig,
                                      indent=4,
                                      sort_keys=True)
        self.configfile = configfile
        
        mainw = QWidget()
        self.setWidget(mainw)
        layout = QGridLayout()
        mainw.setLayout(layout)
        
        self.setWindowTitle('NeuroCAAS configuration')

        tabwidget = QTabWidget()
        layout.addWidget(tabwidget,0,0)

        advancedwid = QWidget()
        lay = QFormLayout()
        advancedwid.setLayout(lay)

        self.cred_access = QLineEdit(self.awsinfo['access_key'])
        lay.addRow(QLabel('AWS access key'),self.cred_access)
        def cred_access():
            self.awsinfo['access_key'] = self.cred_access.text()
        self.cred_access.textChanged.connect(cred_access)

        self.cred_secret = QLineEdit(self.awsinfo['secret_key'])
        lay.addRow(QLabel('AWS secret key'),self.cred_secret)
        def cred_secret():
            self.awsinfo['secret_key'] = self.cred_secret.text()
        self.cred_secret.textChanged.connect(cred_secret)

        self.aws_region = QComboBox()
        for r in awsregions:
            self.aws_region.addItem(r)
        self.aws_region.setCurrentIndex(awsregions.index(self.awsinfo['region']))
        lay.addRow(QLabel('AWS region'),self.aws_region)
        def region_call(value):
            self.awsinfo['region'] = awsregions[value]    
        self.aws_region.currentIndexChanged.connect(region_call)
        
        self.configedit = QPlainTextEdit(ncaasconfig_json)
        lay.addRow(QLabel('NCAAS settings'),self.configedit)
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        lab = QLabel('Log to neurocaas.org and go to the user settings page to get the credentials.')
        l.addWidget(lab)
        web = QWebEngineView()
        l.addWidget(web)
        web.load(QtCore.QUrl("http://www.neurocaas.org/profile/"))
        tabwidget.addTab(w,'NeuroCAAS login')
        tabwidget.addTab(advancedwid,'Settings')

        self.html = ''
        def parsehtml():
            page = web.page()
            def call(var):
                self.html = var
                tt = var.split('\n')
                values = []
                extra = '<input class="form-control" type="text" value="'
                for i,t in enumerate(tt):
                    if extra in t:
                        values.append(t.replace(extra,'').strip(' ').split('"')[0])
                if len(values)>=4:
                    self.cred_access.setText(values[2])
                    self.cred_secret.setText(values[3])
                    print('Got credentials from the website, you can close this window.')
                    dlg = QDialog()
                    dlg.setWindowTitle('Good job!')
                    but = QDialogButtonBox(QDialogButtonBox.Ok)
                    but.accepted.connect(dlg.accept)
                    l = QVBoxLayout()
                    lab = QLabel('Got the credentials from the website, you can now close this window to continue. Or adjust defaults in the Advanced tab')
                    lab.setStyleSheet("font: bold")
                    l.addWidget(lab)
                    l.addWidget(but)
                    dlg.setLayout(l)
                    dlg.exec_()
                    
            page.toHtml(call)
        web.loadFinished.connect(parsehtml)
        #self.getsite = QPushButton('Get credentials from website')
        def setHtml(self,html):
            self.html = html
        #def getsite():
            
        #lay.addRow(self.getsite)
        #self.getsite.setStyleSheet("font: bold")
        #self.getsite.clicked.connect(getsite)
        self.show()
    def closeEvent(self,event):
        ncaas_set_aws_keys(**self.awsinfo)
        print('Saved AWS keys.')
        try:
            from io import StringIO
            pars = json.load(StringIO(self.configedit.toPlainText()))
        except Exception as E:
            print('Error in the configuration file, did not save')
            return
        with open(self.configfile,'w') as fd:
            json.dump(pars,fd,indent=4,sort_keys = True)
        event.accept()
        
class TextEditor(QDockWidget):
    def __init__(self,path,s3=None,bucket=None,
                 parent = None,
                 refresh_interval = 2,watch_file = True):
        super(TextEditor,self).__init__()
        self.parent = parent
        self.s3 = s3
        self.bucketname = bucket
        self.path = path
        self.original = self.refresh_original()
        mainw = QWidget()
        self.setWidget(mainw)
        lay = QVBoxLayout()
        mainw.setLayout(lay)
        self.tx = QTextEdit()
        self.setWindowTitle('NeuroCAAS file - {0}'.format(self.path))
        lay.addWidget(self.tx)
        self.tx.setText(self.original)
        w = QWidget()
        hl = QFormLayout()
        w.setLayout(hl)
        ckbox = QCheckBox()
        hl.addRow(QLabel('Watch file'),ckbox)

        self.timer = QTimer()
        ckbox.setChecked(False)
        def update():
            ori = self.refresh_original()
            if ori == self.original:
                self.tx.setText(self.original)
        self.timer.timeout.connect(update)
        
        def watch(val):
            if val:
                self.timer.start(2000)
            else:
                self.timer.stop()
                print('Timer stopped.')
        ckbox.setChecked(watch_file)
        ckbox.stateChanged.connect(watch)
        lay.addWidget(w)
        
    def refresh_original(self):
        if not self.s3 is None:
            bucket = self.s3.Bucket(self.bucketname)
            bucket.download_file(self.path,tempfile)
            with open(tempfile,'r') as f:
                return f.read()

from boto3.s3.transfer import TransferConfig
GB = 1024 ** 3
# Not being used now.
multipart_config = TransferConfig(multipart_threshold=int(1*GB),
                                  max_concurrency=10,
                                  multipart_chunksize=int(0.1*GB),
                                  use_threads=True)

class  AnalysisSelectionWidget(QDialog):
    def __init__(self, path, config):
        '''
        Select analysis from to be ran on NCAAS
        '''
        super(AnalysisSelectionWidget,self).__init__()
        self.config = config

        # there are 2 parameter sets, one for locaNMF and one for PMD
        
        layout = QGridLayout()
        self.setLayout(layout)
        
        self.setWindowTitle('NeuroCAAS job parameters')
        
        tabwidget = QTabWidget()
        layout.addWidget(tabwidget,0,0)

        selectwid = QWidget()
        lay = QFormLayout()
        selectwid.setLayout(lay)
        
        self.selections = analysis_selection_dict
        def ckchanged(ind,chk):
            val = self.selections[ind]['selected'] = chk.isChecked()
        for i,k in enumerate(self.selections):
            self.selections[i]['checkbox'] = QCheckBox()
            name = QLabel(k['name'])
            lay.addRow(self.selections[i]['checkbox'],name)
            self.selections[i]['checkbox'].clicked.connect(partial(ckchanged, i,
                                                                   self.selections[i]['checkbox']))
            self.selections[i]['checkbox'].setToolTip(self.selections[i]['desc'])
            name.setToolTip(self.selections[i]['desc'])
            self.selections[i]['checkbox'].setChecked(self.selections[i]['selected'])
        tabwidget.addTab(selectwid,'Select analysis')

        parameterswid = QWidget()
        lay = QFormLayout()
        parameterswid.setLayout(lay)
        def vchanged(analysis,option_name,edit):
            dtype = type(self.config[analysis]['config'][option_name])
            self.config[analysis]['config'][option_name] = dtype(edit.text())

        if 'cshl-wfield-preprocessing' in self.config.keys():
            if 'config' in self.config['cshl-wfield-preprocessing'].keys():
                lay.addRow(QLabel('Parameters for motion correction, compression, denoising and hemodynamics compensation'))

                for k in self.config['cshl-wfield-preprocessing']['config'].keys():
                    c = QLineEdit(str(self.config['cshl-wfield-preprocessing']['config'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vchanged,'cshl-wfield-preprocessing' ,k,c))
                    
        if 'cshl-wfield-locanmf' in self.config.keys():
            if 'config' in self.config['cshl-wfield-locanmf'].keys():
                lay.addRow(QLabel('Parameters for feature extraction'))                                  
                for k in self.config['cshl-wfield-locanmf']['config'].keys():
                    c = QLineEdit(str(self.config['cshl-wfield-locanmf']['config'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vchanged,'cshl-wfield-locanmf',k,c))
        tabwidget.addTab(parameterswid,'Parameters')

        files = glob(pjoin(path,'*'))
        fileswid = QWidget()
        lay = QFormLayout()
        fileswid.setLayout(lay)
        
        def filechanged(f,chk):
            val = self.files[f]['selected'] = chk.isChecked()
        files = [dict(fname = f,selected=True) for f in files]
        self.files = []
        for i,fname in enumerate(files):
            if not os.path.isfile(fname['fname']):
                continue
            _,ext = os.path.splitext(fname['fname'])
            if not ext in analysis_extensions:
                fname['selected'] = False
            self.files.append(fname)
            f = QCheckBox()
            name = QLabel(os.path.basename(fname['fname']))
            lay.addRow(f,name)
            f.clicked.connect(partial(filechanged, len(self.files),f))
            f.setChecked(fname['selected'])
        if not len(self.files):
            print('This folder {0} contained no files that can be analysed.'.format(path))
        scroll = QScrollArea()
        scroll.setWidget(fileswid)
        tabwidget.addTab(scroll,'File selection')

        advancedwid = QWidget()
        lay = QFormLayout()
        advancedwid.setLayout(lay)
        def vschanged(analysis,option_name,edit):
            dtype = type(self.config[analysis]['submit'][option_name])
            self.config[analysis]['submit'][option_name] = dtype(edit.text())

        if 'cshl-wfield-preprocessing' in self.config.keys():
            if 'submit' in self.config['cshl-wfield-preprocessing'].keys():
                lay.addRow(QLabel('Advanced parameters for motion correction, compression, denoising and hemodynamics compensation'))
                for k in self.config['cshl-wfield-preprocessing']['submit'].keys():
                    c = QLineEdit(str(self.config['cshl-wfield-preprocessing']['submit'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vschanged,'cshl-wfield-preprocessing' ,k,c))
        if 'cshl-wfield-locanmf' in self.config.keys():
            if 'submit' in self.config['cshl-wfield-locanmf'].keys():
                lay.addRow(QLabel('Advanced parameters for feature extraction'))                                  
                for k in self.config['cshl-wfield-locanmf']['submit'].keys():
                    c = QLineEdit(str(self.config['cshl-wfield-locanmf']['submit'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vschanged,'cshl-wfield-locanmf',k,c))
        tabwidget.addTab(advancedwid,'Advanced configuration')

        
        submit_button = QPushButton('Submit analysis')
        layout.addWidget(submit_button,1,0)
        self.transfer_queue = []
        self.transfer_config = []
        def submit():
            filetransfer = []
            aws_transfer_queue = []
            for f in self.files:
                if f['selected']:
                    filetransfer.append(f['fname'])
            foldername = os.path.basename(path)
            if len(filetransfer):
                aws_transfer_queue.append(
                    dict(name = foldername, #os.path.basename(filetransfer[0]),
                         awsdestination = [foldername + '/inputs/'
                                           +os.path.basename(f) for f in filetransfer],
                         awssubmit = foldername+'/submit.json',
                         awsbucket = None,
                         localpath = filetransfer,
                         last_status = 'pending_transfer'))
            if len(aws_transfer_queue):
                self.transfer_queue = aws_transfer_queue
                self.transfer_config = dict(self.config,selection=self.selections)
            else:
                print('No files selected.')
            self.close()

        submit_button.clicked.connect(submit)    
        self.show()
        
    def closeEvent(self,event):
        event.accept()


def ncaas_upload_queue(path,
                       subfolder = 'inputs',config = None):
    # Search and upload dat file, respect folder structure, return path
    if sys.platform in ['win32']: # path fix on windows
        if path[0] == '/':
            path = path[1:]
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    print('Selected folder: {0}'.format(path))
    print('Choose the analysis to run.')
    dlg = AnalysisSelectionWidget(path,config)
    dlg.exec_()
    return (dlg.transfer_queue,dlg.transfer_config)
    
class NCAASwrapper(QMainWindow):
    def __init__(self,folder = '.',
                 config = pjoin(os.path.expanduser('~'),
                                '.wfield','ncaas_config.json')):
        super(NCAASwrapper,self).__init__()
        self.setWindowTitle('NeuroCAAS preprocessing')
        folder = os.path.abspath(folder)
        if not config is dict: # then it is a filepath
            config = ncaas_read_analysis_config(config)
        self.config = config
        self.folder = folder
        self.uploading = False
        self.fetching_results = False
        
        self.delete_inputs=True
        self.delete_results=True
        
        awskeys = ncaas_read_aws_keys()
        if awskeys['access_key'] == '' or awskeys['secret_key'] == '':
            self.cred = CredentialsManager()
            self.cred.show()
        
        mainw = QWidget()
        self.setCentralWidget(mainw)
        lay = QHBoxLayout()
        mainw.setLayout(lay)
        
        # Filesystem browser
        self.fs_view = FilesystemView(folder, parent=self)
        self.fs_view.expandToDepth(2)
        # Add the widget with label
        w = QWidget()
        l = QFormLayout()
        w.setLayout(l)
        self.folder = QPushButton(folder)
        #self.folder.setStyleSheet("font: bold")
        lab = QLabel('Select local folder:')
        l.addRow(lab, self.folder)
        lab.setStyleSheet("font: bold")
        lab = QLabel('Local folder view - drag to remote:')
        l.addRow(lab)
        lab.setStyleSheet("font: bold")
        ww = QWidget()
        ll = QVBoxLayout()
        ww.setLayout(ll)
        ll.addWidget(w)
        ll.addWidget(self.fs_view)
        lay.addWidget(ww)
        
        def set_folder():
            self.fs_view.query_root()
        self.folder.clicked.connect(set_folder)
        self.open_editors = []        
        # AWS browser
        self.aws_view = AWSView(config, parent=self)
        self.aws_view.expandToDepth(2)

        # Add the widget with label
        historypath = pjoin(os.path.expanduser('~'),
                            '.wfield','ncaas_gui_log.txt')
        self.historyfile = open(historypath,'a+')
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        l.addWidget(QLabel('<b>' + 'NeuroCAAS - {0}'.format(', '.join(self.aws_view.bucketnames) + ' - drop below to upload to the cloud <\b>')))
        l.addWidget(self.aws_view)
        lay.addWidget(w)

        # Update the tranfer q and the log
        self.transferqpath = pjoin(os.path.expanduser('~'),
                            '.wfield','ncaas_transfer_q.json')
        if os.path.isfile(self.transferqpath):
            with open(self.transferqpath,'r') as f:
                try:
                    self.aws_view.aws_transfer_queue = json.load(f)
                except:
                    print('Corrupt transfer queue file?')
        for i,t in enumerate(self.aws_view.aws_transfer_queue):
            if t['last_status'] == 'in_transfer': # There can be no transfers in init - resume
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'pending_transfer'
            if t['last_status'] == 'fetching_results': # there can be no fetching when init - resume
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'submitted'
        
        sw = QWidget()
        sl = QVBoxLayout()
        sw.setLayout(sl)

        self.queuelist = QListWidget()
        self.pbar = QProgressBar()

        self.infomon = QTextEdit()
        self.infomon.insertPlainText(''.join(tail(historypath)))

        self.submitb = QPushButton('Submit to NeuroCAAS')
        self.submitb.setStyleSheet("font: bold")
        
        lab = QLabel('History and neurocaas information:')
        lab.setStyleSheet("font: bold")
        sl.addWidget(lab)
        sl.addWidget(self.infomon)
        lab = QLabel('Job submission queue:')
        lab.setStyleSheet("font: bold")
        sl.addWidget(lab)
        sl.addWidget(self.queuelist)
        sl.addWidget(self.submitb)
        lab = QLabel('Transfer progress:')
        lab.setStyleSheet("font: bold")
        sl.addWidget(lab)
        sl.addWidget(self.pbar)
        l.addWidget(sw)

        self.queuelist.itemDoubleClicked.connect(self.remove_from_queue)
        self.refresh_queuelist()
        self.submitb.clicked.connect(self.process_aws_transfer)
        #import ipdb
        #ipdb.set_trace()
        self.resize(QDesktopWidget().availableGeometry().size() * 0.85);        
        self.show()
        self.fetchresultstimer = QTimer()
        self.fetchresultstimer.timeout.connect(self.fetch_results)
        self.fetchresultstimer.start(3000)
        
    def fetch_results(self):
        '''
        Checks if the data are analyzed and copies the results. 
        '''
        for i,t in enumerate(self.aws_view.aws_transfer_queue):
            resultsdir = os.path.dirname(t['awsdestination'][0]).replace('/inputs',
                                                                         '/results')
            outputsdir = os.path.dirname(t['awsdestination'][0]).replace('/inputs',
                                                                         '/outputs')
            logsdir = os.path.dirname(t['awsdestination'][0]).replace('/inputs',
                                                                      '/logs')
            if t['last_status'] == 'submitted':
                resultsfiles = []
                for a in self.aws_view.awsfiles:
                    if resultsdir in a or outputsdir in a:
                        resultsfiles.append(a)
                if len(resultsfiles):
                    self.aws_view.aws_transfer_queue[i]['last_status'] = 'fetching_results'
                    self.refresh_queuelist()
                    for a in self.aws_view.awsfiles:
                        if logsdir in a:
                            resultsfiles.append(a)
                    print('Found results for {name}'.format(**t))
                    localpath = pjoin(os.path.dirname(t['localpath'][0]),'results')
                    if not os.path.isdir(localpath):
                        os.makedirs(localpath)
                        self.to_log('Creating {0}'.format(localpath))
                    bucket = self.aws_view.s3.Bucket(t['awsbucket'])
                    self.fetching_results = True
                    for f in resultsfiles:
                        def get():
                            # drop the bucket name
                            fn = f.replace(t['awsbucket']+'/','')
                            if outputsdir in f:
                                lf = fn.replace(outputsdir,localpath)
                            if resultsdir in f:
                                lf = fn.replace(resultsdir,localpath)
                            if logsdir in f:
                                lf = fn.replace(logsdir,localpath)
                            if not os.path.isdir(os.path.dirname(lf)):
                                os.makedirs(os.path.dirname(lf))
                            print(fn,lf)
                            bucket.download_file(fn,lf)
                        thread = threading.Thread(target=get)
                        thread.start()
                        self.to_log('Fetching {0}'.format(f))
                        while thread.is_alive():
                            QApplication.processEvents()
                            time.sleep(0.1)
                    self.to_log('Done fetching results to {0}'.format(localpath))

                    self.aws_view.aws_transfer_queue[i]['last_status'] = 'got_results'

                    
                    if self.delete_inputs:
                        # need to delete the remote data
                        configpath = os.path.dirname(t['awsdestination'][0])+'/'+'config.yaml' 
                        submitpath = t['awssubmit']#os.path.dirname(t['awsdestination'])+'/'+'submit.json'
                        for a in t['awsdestination']:
                            self.aws_view.s3.Object(t['awsbucket'],a).delete()
                        self.aws_view.s3.Object(t['awsbucket'],configpath).delete()
                        self.aws_view.s3.Object(t['awsbucket'],submitpath).delete()
                        self.to_log('Remote delete: {0}'.format(t['awsdestination']))
                    if self.delete_results:
                        for f in resultsfiles:
                            f = f.replace(t['awsbucket']+'/','')
                            self.aws_view.s3.Object(t['awsbucket'],f).delete()
                            self.to_log('Remote delete: {0}'.format(f))
                    self.to_log('COMPLETED {0}'.format(t['name']))
                
                    self.fetching_results = False
                    self.remove_from_queue(self.queuelist.item(i))
                    return # because we removed an item from the queue, restart the loop
                    
                    #if self.config['decompress_results']:
                    #    try:
                    #        for f in resultsfiles:
                    #            # read U and decompress this should become a function 
                    #            if 'sparse_spatial.npz' in f:
                    #                fname = pjoin(localpath,'sparse_spatial.npz')
                    #                fcfg =  pjoin(localpath,'config.yaml')
                    #                if os.path.isfile(fcfg):
                    #                    with open(fcfg,'r') as fc:
                    #                        import yaml
                    #                        config = yaml.load(fc)
                    #                        H,W = (config['fov_height'],config['fov_width'])
                    #                    if os.path.isfile(fname):
                    #                        from scipy.sparse import load_npz
                    #                        Us = load_npz(fname)
                    #                        U = np.squeeze(np.asarray(Us.todense()))
                    #                        U = U.reshape([H,W,-1])
                                            # This may overwrite.. prompt
                    #                        np.save(fname.replace('sparse_spatial.npz','U.npy'),U)
                    # #                      self.to_log('Decompressed {0}'.format(f))
                    #                    else:
                    #                        print('Could not decompress (no file)')
                    #                else:
                    #                    print('Could not decompress (no config.yaml?)')
                    #    except Exception as err:
                    #        self.parent.to_log('ERROR: FAILED TO DECOMPRESS. The error was dumped to the console.')
                    #        print(err)


    def remove_from_queue(self,item):
        itemname = item.text()
        names = [t['name'] for t in self.aws_view.aws_transfer_queue]
        #itemname = itemname..strip('Dataset ')split('-')[0].strip(' ')
        ii = None
        for i in range(len(names)):
            if names[i] in itemname:
                ii = i
                break
        #ii = names.index(itemname)
        tt = self.aws_view.aws_transfer_queue.pop(ii)
        self.to_log('Removed {name} from the tranfer queue'.format(**tt))
        self.queuelist.takeItem(self.queuelist.row(item))
        self.store_transfer_queue()
        
    def to_log(self,msg):
        msg  = to_log(msg,logfile = self.historyfile)
        self.infomon.moveCursor(-1)
        self.infomon.insertPlainText(msg)

    def store_transfer_queue(self):
        with open(self.transferqpath,'w') as fp:
            json.dump(self.aws_view.aws_transfer_queue,fp, indent=4)

    def refresh_queuelist(self,store=True):
        if store:
            self.store_transfer_queue()
        for i,itt in enumerate(self.aws_view.aws_transfer_queue):
            if i >= self.queuelist.count():
                it = QListWidgetItem(itt['name'])
                self.queuelist.insertItem(self.queuelist.count(),it)
            item = self.queuelist.item(i)
            item.setText('Dataset {0} - state {1}'.format(itt['name'],itt['last_status']))
            if itt['last_status'] == 'pending_transfer':
                item.setForeground(QColor(204,102,0))
            elif itt['last_status'] in ['in_transfer','fetching_results']:
                item.setForeground(QColor(0,102,0))
            elif itt['last_status'] in ['uploaded']:
                item.setForeground(QColor(255,0,0))
            elif itt['last_status'] == 'submitted':
                item.setForeground(QColor(0,255,0))
            else:
                item.setForeground(QColor(0,0,0))
                

    def process_aws_transfer(self):
        if self.uploading:
            print('Upload in progress')
            return
        self.uploading = True
        self.submitb.setEnabled(False)
        self.submitb.setText('Transfer in progress')                        

        for i,t in enumerate(self.aws_view.aws_transfer_queue):
            print(t)
            if t['last_status'] == 'pending_transfer':
                for it, fname in enumerate(t['localpath']):
                    if os.path.isfile(fname):
                        self.pbar.setValue(0)
                        self.pbar.setMaximum(100)
                        self.aws_view.aws_transfer_queue[i]['last_status'] = 'in_transfer'
                        self.refresh_queuelist()
                        self.count = 0
                        class Upload():
                            def __init__(self,item,config,s3):
                                self.config = config
                                self.s3 = s3
                                self.item = item
                                statinfo = os.stat(fname)
                                self.fsize = statinfo.st_size
                                self.count = 0
                                self.isrunning = False
                            def run(self):
                                def update(chunk):
                                    self.count += chunk
                                    t = self.item
                                    self.isrunning = True
                                bucket =self.s3.Bucket(t['awsbucket'])
                                print('Uploading to {0}'.format(t['awsdestination'][it]))
                                bucket.upload_file(t['localpath'][it],
                                                   t['awsdestination'][it],
                                                   Callback = update)
                                #Config=multipart_config)
                                self.isrunning = False
                        upload = Upload(t,self.config,self.aws_view.s3)
                        self.to_log('Transfering {0}'.format(t['localpath'][it]))
                        thread = threading.Thread(target=upload.run)
                        thread.start()
                        time.sleep(1)
                        cnt = 0
                        while (upload.isrunning):
                            QApplication.processEvents()
                            self.pbar.setValue(np.ceil(upload.count*98/upload.fsize))
                            time.sleep(0.1)
                            cnt+= 1
                            if np.mod(cnt,2) == 0:
                                self.submitb.setStyleSheet("color: red")
                            else:
                                self.submitb.setStyleSheet("color: black")
                            
                        QApplication.processEvents()
                        self.to_log('Done transfering {name}'.format(**t))
                        self.pbar.setMaximum(100)
                    else:
                        self.to_log('File not found {localpath}'.format(**t))
                        self.remove_from_queue(self.queuelist.item(i))
                        self.submitb.setStyleSheet("color: black")
                        self.submitb.setEnabled(False)
                        return
                self.submitb.setStyleSheet("color: black")
                self.pbar.setValue(0)
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'uploaded'
                self.refresh_queuelist()

            if t['last_status'] == 'uploaded':
                # add a config file
                import yaml
                temp = pjoin(os.path.expanduser('~'),'.wfield','temp_config.yaml')
                with open(temp,'w') as f: 
                    yaml.dump(t['config'],f)
                bucket =self.aws_view.s3.Bucket(t['awsbucket'])
                bucket.upload_file(temp,
                                   os.path.dirname(t['awsdestination'][0])+'/'+'config.yaml')
                self.to_log('Uploaded default config to {name}'.format(**t))
                
                temp = pjoin(os.path.expanduser('~'),'.wfield','temp_submit.json')
                t['submit'] = {k:t['submit'][k] for k in t['submit'] if not k in ['userfolder',
                                                                                  'analysis_extension',
                                                                                  'decompress_results']}
                tmp = dict(t['submit'],
                           dataname = os.path.dirname(t['awsdestination'][0])+'/')
                with open(temp,'w') as f:
                    json.dump(tmp, f, indent=4, sort_keys=True)
                bucket.upload_file(temp,
                                   t['awssubmit'])#os.path.dirname(t['awsdestination'][0])+'/'+'submit.json')
                self.to_log('Submitted analysis {name}'.format(**t))
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'submitted'
                        
        self.uploading = False
        self.submitb.setText('Run on NeuroCAAS')                        
        self.submitb.setEnabled(True)

                
        # for f in filetransfer:
        #    fsize = statinfo.st_size

class AWSView(QTreeView):
    def __init__(self,config,parent=None):
        super(AWSView,self).__init__()
        self.parent = parent
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(3)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDropIndicatorShown(True) 
        self.setAutoScroll(True) 

        # add right click support
        
        def menu_here(event):
            self.menu = QMenu(self)
            delete = self.menu.addAction("Delete")
            ii = self.indexAt(event)
            item = self.model.index(ii.row(),0,ii.parent())
            path = get_tree_path([item])
            if path[-1].endswith('submit.json'):
                rerun = self.menu.addAction("Re-submit")
            tmp = self.menu.exec_(self.mapToGlobal(event))
            if tmp is not None:
                if tmp == delete:
                    for p in path:
                        bucketname = p.strip('/').split('/')[0]
                        temp = p.replace(bucketname,'').strip('/')
                        if not self.s3 is None:
                            self.s3.Object(bucketname,temp).delete()
                            print(bucketname,temp)
                elif tmp == rerun:
                    # Download and upload to re-submit. 
                    if not self.s3 is None:
                        for p in path:
                            bucketname = p.strip('/').split('/')[0]
                            fname = p.replace(bucketname,'').strip('/')
                            
                            bucket = self.s3.Bucket(bucketname)
                            bucket.download_file(fname,tempfile)
                            dicttmp = {}
                            dicttmp['awssubmit'] = fname
                            with open(tempfile,'r') as fd:
                                temp = json.load(fd)
                                dicttmp['submit'] = temp
                            dname = temp['dataname']
                            files = []
                            for f in self.awsfiles: # find the datfile
                                if dname in f:
                                    if 'config.yaml' in f:
                                        bucket.download_file(f.replace(bucketname,'').strip('/'),tempfile)
                                        with open(tempfile,'r') as fd:
                                            dicttmp['config'] = yaml.load(fd)
                                        continue
                                    if 'submit.json' in f:
                                        bucket.download_file(f.replace(bucketname,'').strip('/'),tempfile)
                                        with open(tempfile,'r') as fd:
                                            dicttmp['submit'] = json.load(fd)
                                        continue
                                    files.append(f.replace(bucketname,'').strip('/'))
                                    if 'inputs' in f:
                                        tt = f.split('/')
                                        localpath = None
                                        for i,t in enumerate(tt): # todo: re-write
                                            if t == 'inputs':
                                                localpath = os.path.abspath(pjoin(os.path.curdir,*tt[i+1:-1]))
                                        if localpath is None:
                                            self.to_log('Could not set local folder to re-submit.')
                                        localpath = os.path.curdir
                                    
                            if len(files):
                                toadd = dict(dicttmp, name = os.path.basename(files[0]),
                                             awsdestination = files,
                                             awsbucket=bucketname,
                                             localpath = [pjoin(localpath,os.path.basename(files[0]))],
                                             last_status = "uploaded")
                                self.aws_transfer_queue.append(toadd)
                                self.parent.refresh_queuelist()
                                self.parent.to_log('Re-submitted {0} to the queue, (press "Run on NCAAS" to run).'.format(path))
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(menu_here)
        
        self.config = config
        self.s3 = s3_connect()
        print('Connected to Amazon Web Services')
        self.bucketnames = self.config.keys()
        print('Using buckets: {0}'.format(', '.join(self.bucketnames)))
        self.awsfiles = []
        self.aws_transfer_queue = []

        self.model = AWSItemModel()
        self.setModel(self.model)
        #aws_model.setEditable(False)
        def open_file(value):
            paths = get_tree_path([value])
            extension = os.path.splitext(paths[0])[-1]
            if extension in ['.yaml','.txt','.json']:
                bucket = paths[0].strip('/').split('/')[0]
                temp = paths[0].replace(bucket,'').strip('/')
                wid = TextEditor(temp, s3=self.s3, bucket = bucket)
                self.parent.addDockWidget(Qt.RightDockWidgetArea, wid)
                wid.setFloating(True)
                self.parent.open_editors.append(paths[0])
            elif extension == '':
                print('Folder: {0}'.format(paths[0]))
        self.doubleClicked.connect(open_file)
        self.update_files()
        # These cause refresh, need to check if there are new files first.
        self.timer_update = QTimer()
        self.timer_update.timeout.connect(self.update_files)
        self.timer_update.start(1500)

    def update_files(self):
        awsfiles = s3_ls(self.s3,self.bucketnames)
        if len(awsfiles) == len(self.awsfiles):
            return
        self.awsfiles = awsfiles
        self.model.clear()
        self.model.setHorizontalHeaderLabels([', '.join(self.bucketnames)])
        root = QStandardItem()
        filetree = {}
        [make_tree(i.split("/"), filetree) for i in self.awsfiles]
        build_tree(filetree,root)
        self.model.appendRow(root)
        #index = self.aws_model.indexFromItem(root)
        self.expandAll()
        
    def dragEnterEvent(self, e):        
        if e.mimeData().hasUrls():
            self.setSelectionMode(1)
            e.accept()
        else:
            e.ignore()
    def dragMoveEvent(self, e):
        item = self.indexAt(e.pos())
        self.setCurrentIndex(item)
        e.accept()

    def dragLeaveEvent(self, e):
        self.setSelectionMode(3)
        e.accept()
        
    def dropEvent(self, e):
        paths = get_tree_path([self.indexAt(e.pos())])
        self.setSelectionMode(3)
        [print('Selecting from: {1} to aws {0}'.format(
            paths[0],
            p.path())) for p in e.mimeData().urls()]
        for p in e.mimeData().urls():
            path = p.path()
            transfer, configselection = ncaas_upload_queue(
                path,
                subfolder = 'inputs', config = self.config)
            # this is where we parse the bucket.
            if len(configselection) == 0:
                print('Nothing selected')
                continue
            selection = configselection['selection']
            if (selection[0]['selected'] or # motion
                selection[1]['selected'] or # compression
                selection[2]['selected']):  # hemodynamics
                bucketname = 'cshl-wfield-preprocessing' # TODO: this should not be hardcoded
            elif selection[3]['selected']:
                bucketname = 'cshl-wfield-locanmf' # TODO: this should not be hardcoded
            else:
                print('No analysis were selected ? ')
            print('Running analysis on {0}'.format(bucketname))
            
            for t in transfer:
                print('Placing {0} in the transfer queue.'.format(t['name']))
                t['awsbucket'] = bucketname
                t['config'] = configselection[bucketname]['config']
                t['submit'] = configselection[bucketname]['submit']
                t['awssubmit'] = t['submit']['userfolder']+'/'+t['awssubmit']
                for i,f in enumerate(t['awsdestination']):
                    t['awsdestination'][i] = t['submit']['userfolder']+'/'+f

                # Check if it will run locaNMF
                t['selection'] = [s['acronym'] for s in configselection['selection'] if s['selected']]
                if 'locanmf' in t['selection']:
                    t['locanmf_config'] = configselection['cshl-wfield-locanmf']['config'] 
                    t['locanmf_submit'] = configselection['cshl-wfield-locanmf']['submit'] 
                added = False
                if len(self.aws_transfer_queue): # check if it is already there
                    names = [a['name'] for a in self.aws_transfer_queue]
                    if not t['name'] in names:
                        added = True
                        self.aws_transfer_queue.append(t)
                    else:
                        self.parent.to_log('{name} was already in the transfer queue'.format(**t))
                else:
                    added = True
                    self.aws_transfer_queue.append(t)
            if added:
                self.parent.to_log('Added {name} to transfer queue'.format(**t))
            self.parent.refresh_queuelist()
        e.ignore() # Dont drop the remote table
        
class AWSItemModel(QStandardItemModel):
    def __init__(self):
        super(AWSItemModel,self).__init__()

    def mimeData(self,idx):
        tt = QMimeData()
        tt.setText(','.join(get_tree_path(idx)))
        return tt
        
class FilesystemView(QTreeView):
    def __init__(self,folder,parent=None):
        super(FilesystemView,self).__init__()
        self.parent = parent
        self.fs_model = QFileSystemModel(self)
        self.fs_model.setReadOnly(True)
        self.setModel(self.fs_model)
        self.folder = folder
        self.setRootIndex(self.fs_model.setRootPath(folder))
        self.fs_model.removeColumn(1)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(3)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDropIndicatorShown(True)
        #[self.hideColumn(i) for i in range(1,4)]
        self.setColumnWidth(0,self.width()*.7)
    def query_root(self):
        folder = QFileDialog().getExistingDirectory(self,"Select directory",os.path.curdir)
        self.setRootIndex(self.fs_model.setRootPath(folder))
        self.expandAll()
        self.folder = folder
        if hasattr(self.parent,'folder'):
            self.parent.folder.setText('{0}'.format(folder))

    def dragEnterEvent(self, e):
        print(e)
        item = self.indexAt(e.pos())
        if e.mimeData().hasText():
            self.setSelectionMode(1)
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        item = self.indexAt(e.pos())
        self.setCurrentIndex(item)
        e.accept()

    def dragLeaveEvent(self, e):
        self.setSelectionMode(3)
        e.accept()
        
    def dropEvent(self, e):
        idx = self.model().index(
            self.indexAt(e.pos()).row(),
            0,self.indexAt(e.pos()).parent())
        paths = get_tree_path([idx])
        to_fetch = e.mimeData().text()
        
        files = []
        for a in self.parent.aws_view.awsfiles:
            if to_fetch in a:
                files.append(a)
        if not len(files):
            print('No files listed.')
            e.ignore()
            return
        path = [pjoin(self.folder,p) for p in paths][0]
        path = os.path.abspath(path)
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if len(files)>1:
            # Then there are multiple files, it must be a folder
            localpath = pjoin(path,os.path.basename(to_fetch))
        else:
            localpath = path
        if not os.path.isdir(localpath):
            os.makedirs(localpath)
        self.parent.to_log('Fetching to {0}'.format(localpath))

        for f in files:

            bucketname = f.strip('/').split('/')[0]
            f = f.replace(bucketname,'').strip('/')
            bucket = self.parent.aws_view.s3.Bucket(bucketname)
            def get():
                bucket.download_file(f,pjoin(localpath,os.path.basename(f)))
            thread = threading.Thread(target=get)
            thread.start()
            self.parent.to_log('MANUAL COPY {0}'.format(f))
            while thread.is_alive():
                QApplication.processEvents()
                time.sleep(0.1)
        self.parent.to_log('Done fetching results to {0}'.format(localpath))
        #if self.parent.config['decompress_results']:
            #try:
            #    for f in files:
                    # read U and decompress
                    # this should become a function 
                    #if 'sparse_spatial.npz' in f:
                    #    fname = pjoin(localpath,'sparse_spatial.npz')
                    #    fcfg =  pjoin(localpath,'config.yaml')
                    #    if os.path.isfile(fcfg):
                    #        with open(fcfg,'r') as fc:
                    #            config = yaml.load(fc)
                    #            H,W = (config['fov_height'],config['fov_width'])
                    #        if os.path.isfile(fname):
                    #            from scipy.sparse import load_npz
                    #            Us = load_npz(fname)
                    #            U = np.squeeze(np.asarray(Us.todense()))
                    #            U = U.reshape([H,W,-1])
                    #            np.save(fname.replace('sparse_spatial.npz','U.npy'),U)
                    #            self.parent.to_log('Decompressed {0}'.format(f))
                    #        else:
                    #            print('Could not decompress (no file)')
                    #    else:
                    #        print('Could not decompress (no config.yaml?)')
            #except Exception as err:
            #    self.parent.to_log('ERROR: FAILED TO DECOMPRESS. The error was dumped to the console.')
            #    print(err)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Do you want to delete the remote files?")
        msg.setInformativeText("Do you want to delete remote files:")
        msg.setWindowTitle("[NCAAS] Remote delete?")
        msg.setDetailedText("The following files will be deleted: "+'\n'.join(files))
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        delete_files = dict(msg = False)
        def msgboxok(b):
            if 'OK' in b.text():
                delete_files['msg'] = True
            else:
                print(b.text())
        msg.buttonClicked.connect(msgboxok)
        msg.exec_()
        delete_files = delete_files['msg']

        if delete_files:
            # need to delete the remote data
            for f in files:
                self.parent.aws_view.s3.Object(bucketname,f).delete()
                self.parent.to_log('Remote delete: {0}'.format(f))

        self.parent.to_log('FINISHED MANUAL COPY {0}'.format(to_fetch))
        self.setSelectionMode(3)
        e.ignore()
    
def get_tree_path(items,root = ''):
    ''' Get the paths from a QTreeView item'''
    paths = []
    for item in items:
        level = 0
        index = item
        paths.append([index.data()])
        while index.parent().isValid():
            index = index.parent()
            level += 1
            paths[-1].append(index.data())
        for i,p in enumerate(paths[-1]):
            if p is None :
                paths[-1][i] = ''
        paths[-1] = '/'.join(paths[-1][::-1])
    return paths

def main(folder = '.'):
    if QApplication.instance() != None:
        app = QApplication.instance()
    else:
        app = QApplication(sys.argv)
    awskeys = ncaas_read_aws_keys()
    if awskeys['access_key'] == '' or awskeys['secret_key'] == '':
        print('NeuroCAAS credentials not found.')
        cred = CredentialsManager()
        app.exec_()
        awskeys = ncaas_read_aws_keys()
    from botocore.exceptions import ClientError
    try:
        wind = NCAASwrapper(folder = folder)
        sys.exit(app.exec_())
        #s3_connect()
    except ClientError:
        print('Could not connect to NeuroCAAS, check credentials.')
        cred = CredentialsManager()
        app.exec_()
        awskeys = ncaas_read_aws_keys()
        wind = NCAASwrapper(folder = folder)
        sys.exit(app.exec_())
