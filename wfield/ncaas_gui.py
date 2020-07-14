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
                             QLineEdit,
                             QCheckBox,
                             QComboBox,
                             QListWidget,
                             QLabel,
                             QProgressBar,
                             QFileDialog,
                             QDesktopWidget,
                             QListWidgetItem,
                             QFileSystemModel,
                             QAbstractItemView,
                             QMenu, QAction)

from PyQt5 import QtCore
from PyQt5.QtGui import QStandardItem, QStandardItemModel,QColor
from PyQt5.QtCore import Qt, QTimer,QMimeData

try:
    import boto3
except:
    print('boto3 not installed, installing with pip ')
    from subprocess import call
    call('pip install boto3',shell = True)
    import boto3


tempfile = pjoin(os.path.expanduser('~'),'.wfield','tempfile')

defaultconfig = dict(analysis = 'cshl-wfield-preprocessing',
                     userfolder = 'ChurchlandLab',
                     instance_type =  'r5.16xlarge',
                     analysis_extension = '.dat',
                     config = dict(block_height = 90,
                                   block_width = 80,
                                   frame_rate = 30,
                                   max_components = 15,
                                   num_sims = 64,
                                   overlapping = True,
                                   window_length = 7200))

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
    return config

def s3_connect():
    return boto3.resource('s3')

def s3_ls(s3,bucketname):
    bucket = s3.Bucket(bucketname)
    return [l.key for l in list(bucket.objects.all())]
    
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
        super(CredentialsManager,self).__init__()
        self.awsinfo = ncaas_read_aws_keys()
        self.ncaasconfig = ncaas_read_analysis_config(configfile)
        ncaasconfig_json = json.dumps(self.ncaasconfig,
                                      indent=4,
                                      sort_keys=True)
        mainw = QWidget()
        self.setWidget(mainw)
        lay = QFormLayout()
        mainw.setLayout(lay)
        self.setWindowTitle('NeuroCAAS configuration')

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
        
        self.configedit = QTextEdit(ncaasconfig_json)
        lay.addRow(QLabel('NCAAS settings'),self.configedit)
        self.save = QPushButton('Save')
        def save():
            
            ncaas_set_aws_keys(**self.awsinfo)
            print('Saved AWS keys.')
            try:
                from io import StringIO
                pars = json.load(StringIO(self.configedit.toPlainText()))
            except Exception as E:
                print('Error in the configuration file, did not save')
                return
            with open(configfile,'w') as fd:
                json.dump(pars,fd,indent=4,sort_keys = True)
        lay.addRow(self.save)
        self.save.setStyleSheet("font: bold")
        self.save.clicked.connect(save)
        self.show()
        
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
def ncaas_dat_upload_queue(path,
                           analysis,
                           userfolder,
                           subfolder = 'inputs',config = None):
    # Search and upload dat file, respect folder structure, return path 
    if os.path.isfile(path):
        path = os.path.dirname(path)
    foldername = os.path.basename(path)
    if sys.platform in ['win32']: # path fix on windows
        if path[0] == '/':
            path = path[1:]
    filetransfer = []
    aws_transfer_queue = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if config['analysis_extension'] in f:
                filetransfer.append(pjoin(path,f))
                aws_transfer_queue.append(
                    dict(name = os.path.basename(f),
                         awsdestination = userfolder+'/inputs/'+foldername+'/'+os.path.basename(f),
                         localpath = pjoin(root,f),
                         last_status = 'pending_transfer'))
    return aws_transfer_queue
    
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
        
        self.delete_inputs=False
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
        self.fs_view = FilesystemView(folder,parent=self)
        self.fs_view.expandToDepth(2)
        # Add the widget with label
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        self.folder = QPushButton(folder)
        self.folder.setStyleSheet("font: bold")
        l.addWidget(self.folder)
        l.addWidget(self.fs_view)
        lay.addWidget(w)
        def set_folder():
            self.fs_view.query_root()
        self.folder.clicked.connect(set_folder)
        self.open_editors = []        
        # AWS browser
        self.aws_view = AWSView(config,parent=self)
        self.aws_view.expandToDepth(2)

        # Add the widget with label
        historypath = pjoin(os.path.expanduser('~'),
                            '.wfield','ncaas_gui_log.txt')
        self.historyfile = open(historypath,'a+')
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        bucketname = '/'.join([self.config['analysis'],
                               self.config['userfolder']])
        l.addWidget(QLabel('<b>' + 'NeuroCAAS - {0}'.format(bucketname) + '<\b>'))
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
        sl = QGridLayout()
        sw.setLayout(sl)
        self.queuelist = QListWidget()
        self.pbar = QProgressBar()

        self.infomon = QTextEdit()
        self.infomon.insertPlainText(''.join(tail(historypath)))

        self.submitb = QPushButton('Run on NCAAS')
        self.submitb.setStyleSheet("font: bold")

        sl.addWidget(self.queuelist,1,0,2,1)        
        sl.addWidget(self.infomon,0,0,1,2)        
        sl.addWidget(self.submitb,1,1,1,1)
        sl.addWidget(self.pbar,2,1,1,1)
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
            resultsdir = os.path.dirname(t['awsdestination']).replace('{0}/inputs'.format(self.config['userfolder']),
                                                                      '{0}/results'.format(self.config['userfolder']))
            logsdir = os.path.dirname(t['awsdestination']).replace('{0}/inputs'.format(self.config['userfolder']),
                                                                   '{0}/logs'.format(self.config['userfolder']))
            if t['last_status'] == 'submitted':
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'fetching_results'
                self.refresh_queuelist()
                resultsfiles = []
                for a in self.aws_view.awsfiles:
                    if resultsdir in a:
                        resultsfiles.append(a)
                if len(resultsfiles):
                    if logsdir in a:
                        resultsfiles.append(a)
                    print('Found results for {name}'.format(**t))
                    localpath = pjoin(os.path.dirname(t['localpath']),'results')
                    if not os.path.isdir(localpath):
                        os.makedirs(localpath)
                        self.to_log('Creating {0}'.format(localpath))
                    bucket = self.aws_view.s3.Bucket(self.config["analysis"])
                    self.fetching_results = True
                    for f in resultsfiles:
                        def get():
                            bucket.download_file(f,pjoin(localpath,os.path.basename(f)))
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
                        configpath = os.path.dirname(t['awsdestination'])+'/'+'config.yaml' 
                        submitpath = os.path.dirname(t['awsdestination'])+'/'+'submit.json'
                        self.aws_view.s3.Object(self.config["analysis"],t['awsdestination']).delete()
                        self.aws_view.s3.Object(self.config["analysis"],configpath).delete()
                        self.aws_view.s3.Object(self.config["analysis"],submitpath).delete()
                        self.to_log('Remote delete: {0}'.format(t['awsdestination']))
                    if self.delete_results:
                        for f in resultsfiles:
                            self.aws_view.s3.Object(self.config["analysis"],f).delete()
                            self.to_log('Remote delete: {0}'.format(f))
                    self.to_log('COMPLETED {0}'.format(t['name']))
                    self.fetching_results = False
                    self.remove_from_queue(self.queuelist.item(i))
                    return # because we removed an item from the queue, restart the loop

    def remove_from_queue(self,item):
        itemname = item.text()
        names = [t['name'] for t in self.aws_view.aws_transfer_queue]
        ii = names.index(itemname)
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
            json.dump(self.aws_view.aws_transfer_queue,fp,indent=4)

    def refresh_queuelist(self,store=True):
        if store:
            self.store_transfer_queue()
        for i,itt in enumerate(self.aws_view.aws_transfer_queue):
            if i >= self.queuelist.count():
                it = QListWidgetItem(itt['name'])
                self.queuelist.insertItem(self.queuelist.count(),it)
            item = self.queuelist.item(i)
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
        for i,t in enumerate(self.aws_view.aws_transfer_queue):
            print(t)
            if t['last_status'] == 'pending_transfer':
                if os.path.isfile(t['localpath']):
                    self.aws_view.aws_transfer_queue[i]['last_status'] = 'in_transfer'
                    self.refresh_queuelist()
                    self.pbar.setValue(0)
                    self.pbar.setMaximum(100)
                    self.count = 0
                    class Upload():
                        def __init__(self,item,config,s3):
                            self.config = config
                            self.s3 = s3
                            self.item = item
                            statinfo = os.stat(t['localpath'])
                            self.fsize = statinfo.st_size
                            self.count = 0
                            self.isrunning = False
                        def run(self):
                            def update(chunk):
                                self.count += chunk
                            t = self.item
                            self.isrunning = True
                            bucket =self.s3.Bucket(self.config['analysis'])
                            print('Uploading to {0}'.format(t['awsdestination']))
                            bucket.upload_file(t['localpath'],
                                               t['awsdestination'],
                                               Callback = update)
                                               #Config=multipart_config)
                            self.isrunning = False
                    upload = Upload(t,self.config,self.aws_view.s3)
                    self.to_log('Transfering {name}'.format(**t))
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
                            
                    self.submitb.setStyleSheet("color: red")
                    QApplication.processEvents()
                    self.to_log('Done transfering {name}'.format(**t))
                    self.aws_view.aws_transfer_queue[i]['last_status'] = 'uploaded'
                else:
                    self.to_log('File not found {localpath}'.format(**t))
                    self.remove_from_queue(self.queuelist.item(i))
                    return
            if t['last_status'] == 'uploaded':
                # add a config file
                import yaml
                tempfile = pjoin(os.path.expanduser('~'),'.wfield','temp_config.yaml')
                with open(tempfile,'w') as f: 
                    yaml.dump(self.config['config'],f)
                bucket =self.aws_view.s3.Bucket(self.config['analysis'])
                bucket.upload_file(tempfile,
                                   os.path.dirname(t['awsdestination'])+'/'+'config.yaml')
                self.to_log('Uploaded default config to {name}'.format(**t))
                
                tempfile = pjoin(os.path.expanduser('~'),'.wfield','temp_submit.json')
                tmp = dict(dataname = os.path.dirname(t['awsdestination']),
                           instance_type =  self.config["instance_type"])
                with open(tempfile,'w') as f:
                    json.dump(tmp,f)
                bucket =self.aws_view.s3.Bucket(self.config['analysis'])
                bucket.upload_file(tempfile,
                                   os.path.dirname(t['awsdestination'])+'/'+'submit.json')
                self.to_log('Submitted analysis {name}'.format(**t))
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'submitted'
                        
        self.uploading = False

                
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
            path = get_tree_path([item])[0]
            if path.endswith('submit.json'):
                rerun = self.menu.addAction("Re-submit")
            tmp = self.menu.exec_(self.mapToGlobal(event))
            if tmp is not None:
                if tmp == delete:
                    path = get_tree_path([item])[0]
                    self.s3.Object(self.config["analysis"],path).delete()
                elif tmp == rerun:
                    # Download and upload to re-submit. 
                    if not self.s3 is None:
                        
                        bucket = self.s3.Bucket(self.bucketname)
                        bucket.download_file(path,tempfile)
                        with open(tempfile,'r') as fd:
                            temp = json.load(fd)
                        dname = temp['dataname']
                        for f in self.awsfiles: # find the datfile
                            if dname in f and f.endswith(self.config['analysis_extension']):
                                if 'inputs' in f:
                                    tt = f.split('/')
                                    localpath = None
                                    for i,t in enumerate(tt): # todo: re-write
                                        if t == 'inputs':
                                            localpath = os.path.abspath(pjoin(os.path.curdir,*tt[i+1:-1]))
                                    if localpath is None:
                                        self.to_log('Could not set local folder to re-submit.')
                                        localpath = os.path.curdir
                                    toadd = dict(name = os.path.basename(f),
                                                 awsdestination = f,
                                                 localpath = pjoin(localpath,os.path.basename(f)),
                                                 last_status = "uploaded")
                                    self.aws_transfer_queue.append(toadd)
                                    self.parent.refresh_queuelist()
                                    self.parent.to_log(
                                        'Re-submitted {0} to the queue, (press "Run on NCAAS" to run).'.format(path))
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(menu_here)
        
        self.config = config
        self.s3 = s3_connect()
        self.bucketname = self.config['analysis']
        self.awsfiles = []
        self.aws_transfer_queue = []

        self.model = AWSItemModel()
        self.setModel(self.model)
        self.model.setHorizontalHeaderLabels([self.config['analysis']])        
        #aws_model.setEditable(False)
        def open_file(value):
            paths = get_tree_path([value])
            extension = os.path.splitext(paths[0])[-1]
            if extension in ['.yaml','.txt','.json']:
                wid = TextEditor(paths[0],s3=self.s3,bucket = self.bucketname)
                self.parent.addDockWidget(Qt.RightDockWidgetArea,wid)
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
        awsfiles = s3_ls(self.s3,self.bucketname)
        if len(awsfiles) == len(self.awsfiles):
            return
        self.awsfiles = awsfiles
        self.model.clear()
        self.model.setHorizontalHeaderLabels([self.config['analysis']])        
        root = QStandardItem(self.config['userfolder'])
        filetree = {}
        [make_tree(i.split("/"), filetree) for i in self.awsfiles]
        build_tree(filetree[self.config['userfolder']],root)
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
        [print('Goint to  upload from: {1} to aws {0}'.format(
            paths[0],
            p.path())) for p in e.mimeData().urls()]
        for p in e.mimeData().urls():
            path = p.path()
            tt = ncaas_dat_upload_queue(
                path,
                analysis = self.config['analysis'],
                userfolder = self.config['userfolder'],
                subfolder = 'inputs',config = self.config)
            for t in tt:
                if len(self.aws_transfer_queue):
                    names = [a['name'] for a in self.aws_transfer_queue]
                    print(names)
                    print(t['name'])
                    if not t['name'] in names:
                        self.aws_transfer_queue.append(t)
                        self.parent.to_log('Added {name} to transfer queue'.format(**t))
                    else:
                        self.parent.to_log('{name} was already in the transfer queue'.format(**t))
                else:
                    self.aws_transfer_queue.append(t)
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
    for i,p in enumerate(paths):
        if None in p:
            paths[i] = ['']
    return ['/'.join(p[::-1]) for p in paths]

def main():
    if QApplication.instance() != None:
        app = QApplication.instance()
    else:
        app = QApplication(sys.argv)
    awskeys = ncaas_read_aws_keys()
    if awskeys['access_key'] == '' or awskeys['secret_key'] == '':
        print('NeuroCAAS credentials not found.')
        cred = CredentialsManager()
        app.exec_()
    try:
        s3_connect()
    except Exception as E:
        print('Could not connect to NeuroCAAS, check credentials.')
        sys.exit()
        
    wind = NCAASwrapper(folder = '.')
    sys.exit(app.exec_())
