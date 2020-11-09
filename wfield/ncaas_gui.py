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

from .ncaas_utils import *
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

PMD_BUCKET = 'cshl-wfield-preprocessing'
NMF_BUCKET = 'cshl-wfield-locanmf'

tempfile = pjoin(os.path.expanduser('~'),'.wfield','tempfile')

analysis_extensions = ['.bin','.tiff','.tif','.npy','.json','.dat']
ZIP_PMD = True

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
                    but = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                    def ok():
                        self.close()
                        dlg.accept()
                    but.accepted.connect(ok)
                    but.rejected.connect(dlg.accept)
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
        ckbox.setChecked(watch_file)
        def update():
            ori = self.refresh_original()
            if not ori == self.original:
                self.original = ori
                self.tx.setText(self.original)
        self.timer.timeout.connect(update)
        if watch_file:
            self.timer.start(2000)
        
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

    def closeEvent(self,evt):
        self.timer.stop()
        print('Stopped timer file: {0}'.format(self.path))
        evt.accept()

class  AnalysisSelectionWidget(QDialog):
    def __init__(self, path, config):
        '''
        Select analysis from to be ran on NCAAS.

        '''
        super(AnalysisSelectionWidget,self).__init__()
        self.config = dict(**config)

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
            if ind == 1:
                self.selections[2]['checkbox'].setChecked(False)
                self.selections[2]['selected'] = False
                print('Hemodynamics correction needs a compressed dataset.')
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
            try:
                self.config[analysis]['config'][option_name] = dtype(edit.text())
                if option_name ==  'num_channels':
                    if not self.config[analysis]['config'][option_name] == 2:
                        #disable hemodynamics
                        self.selections[2]['checkbox'].setChecked(False)
                        self.selections[2]['checkbox'].setEnabled(False)
                        self.selections[2]['selected'] = False
                        print('WARNING: Disabled hemodynamics correction because of num_channels')
                    else:
                        self.selections[2]['checkbox'].setChecked(True)
                        self.selections[2]['selected'] = True
                        self.selections[2]['checkbox'].setEnabled(True)
                        print('WARNING: Enabled hemodynamics correction because of num_channels = 2')
            except:
                print('Could not get {0}.'.format(option_name))
        if PMD_BUCKET in self.config.keys():
            if 'config' in self.config[PMD_BUCKET].keys():
                lay.addRow(QLabel('Parameters for motion correction, compression, denoising and hemodynamics compensation'))

                for k in self.config[PMD_BUCKET]['config'].keys():
                    if not 'analysis_selection' == k:
                        c = QLineEdit(str(self.config[PMD_BUCKET]['config'][k]))
                        name = QLabel(k)
                        lay.addRow(name,c)
                        c.textChanged.connect(partial(vchanged,PMD_BUCKET ,k,c))
                    
        if NMF_BUCKET in self.config.keys():
            if 'config' in self.config[NMF_BUCKET].keys():
                lay.addRow(QLabel('Parameters for feature extraction'))                                  
                for k in self.config[NMF_BUCKET]['config'].keys():
                    c = QLineEdit(str(self.config[NMF_BUCKET]['config'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vchanged,NMF_BUCKET,k,c))
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
            f.clicked.connect(partial(filechanged, len(self.files)-1,f))
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
            try:
                self.config[analysis]['submit'][option_name] = dtype(edit.text())
            except:
                print('Could not get {0}'.format(option_name))

        if PMD_BUCKET in self.config.keys():
            if 'submit' in self.config[PMD_BUCKET].keys():
                lay.addRow(QLabel('Advanced parameters for motion correction, compression, denoising and hemodynamics compensation'))
                for k in self.config[PMD_BUCKET]['submit'].keys():
                    c = QLineEdit(str(self.config[PMD_BUCKET]['submit'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vschanged,PMD_BUCKET ,k,c))
        if NMF_BUCKET in self.config.keys():
            if 'submit' in self.config[NMF_BUCKET].keys():
                lay.addRow(QLabel('Advanced parameters for feature extraction'))                                  
                for k in self.config[NMF_BUCKET]['submit'].keys():
                    c = QLineEdit(str(self.config[NMF_BUCKET]['submit'][k]))
                    name = QLabel(k)
                    lay.addRow(name,c)
                    c.textChanged.connect(partial(vschanged,NMF_BUCKET,k,c))
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
                         awsdestination = ['{0}' + foldername + '{1}'
                                           +os.path.basename(f) for f in filetransfer],
                         awssubmit = '{0}'+foldername+'{1}submit.json',
                         awsbucket = None,
                         localpath = filetransfer,
                         last_status = 'pending_transfer',
                         last_change_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
            if len(aws_transfer_queue):
                self.transfer_queue = aws_transfer_queue
                self.transfer_config = dict(self.config, selection=self.selections)
                if self.selections[3]['selected']:
                    # Then locaNMF is selected.
                    lmarkfiles = glob(pjoin(path,'*_landmarks.json'))
                    if not len(lmarkfiles):
                        print('Opening the dataset to compute get Allen Landmarks.')
                        from .io import load_stack
                        from .widgets import AllenMatchWidget
                        dat = load_stack(path)
                        dlg = QDialog()
                        dlg.setWindowTitle('Allen alignment is required for locaNMF')
                        l = QVBoxLayout()
                        lab = QLabel('Move the point to the respective landmarks and close this window.')
                        lab.setStyleSheet("font: bold")
                        l.addWidget(lab)
                        w = AllenMatchWidget(raw = dat,
                                             folder = path)
                        l.addWidget(w)
                        dlg.setLayout(l)
                        dlg.exec_()
                    else:
                        print('Found a landmarks file.')
                    print(path)
            else:
                print('No files selected.')
            self.close()

        submit_button.clicked.connect(submit)    
        self.show()
        
    def closeEvent(self,event):
        event.accept()


def ncaas_upload_queue(path,config = None):
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
        if self.uploading:
            return
        
        for i,t in enumerate(self.aws_view.aws_transfer_queue):
            resultsdir = os.path.dirname(t['awsdestination'][0]).replace('/inputs',
                                                                         '/results')
            outputsdir = os.path.dirname(t['awsdestination'][0]).replace('/inputs',
                                                                         '/outputs')
            logsdir = os.path.dirname(t['awsdestination'][0]).replace('/inputs',
                                                                      '/logs')
            if t['last_status'] == 'submitted':
                resultsfiles = []
                awsfiles = s3_ls(self.aws_view.s3,self.aws_view.bucketnames) # update directly here (no lag).
                for a in awsfiles:
                    if resultsdir in a or outputsdir in a:
                        resultsfiles.append(a)
                if len(resultsfiles):
                    # check that the results are all there (this should change in the next implementation of neurocaas, hopefully)
                    if t['awsbucket'] == PMD_BUCKET:
                        got_all_results = [False,False]
                        for j,res in enumerate(['reduced_spatial.npy',
                                                'reduced_temporal.npy']):
                            for f in resultsfiles:
                                if res in f:
                                    got_all_results[j] = True
                        got_all_results = np.sum(got_all_results) == 2
                    else:
                        got_all_results = True # Don't bother doing this for locaNMF?
                    if got_all_results == False:
                        print('Not all files were there for the PMD bucket?')
                        return # Do nothing
                    self.submitb.setEnabled(False)
                    self.submitb.setText('Downloading result files')                        

                    self.aws_view.aws_transfer_queue[i]['last_status'] = 'fetching_results'
                    self.aws_view.aws_transfer_queue[i]['last_change_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    self.refresh_queuelist()
                    for a in self.aws_view.awsfiles:
                        if logsdir in a:
                            resultsfiles.append(a)
                    print('Found results for {name}'.format(**t))
                    localpath = pjoin(os.path.dirname(t['localpath'][0]),'results')
                    localpath = os.path.abspath(localpath)
                    if not os.path.isdir(localpath):
                        os.makedirs(localpath)
                        self.to_log('Creating {0}'.format(localpath))
                    bucket = self.aws_view.s3.Bucket(t['awsbucket'])
                    self.fetching_results = True
                    safe_delete = True
                    t['downloaded'] = []
                    t['safe_delete'] = True
                    def get():
                        for f in resultsfiles:
                            # drop the bucket name
                            fn = f.replace(t['awsbucket']+'/','')
                            if outputsdir in f:
                                lf = fn.replace(outputsdir,localpath)
                            if resultsdir in f:
                                lf = fn.replace(resultsdir,localpath)
                            if logsdir in f:
                                lf = fn.replace(logsdir,localpath)
                            lf = lf.split('/')
                            lf = pjoin(*lf)
                            if not os.path.isdir(os.path.dirname(lf)):
                                os.makedirs(os.path.dirname(lf))
                            print(fn,lf)
                            t['downloaded'].append(lf)
                            if not os.path.isdir(lf):
                                try:
                                    bucket.download_file(fn,lf)
                                except:
                                    print('Warning: Could not download the file {0}. Is it open?'.format(lf))
                                    t['safe_delete'] = False
                                    
                            self.to_log('Fetching {0}'.format(lf))
                    safe_delete=t['safe_delete']
                    thread = threading.Thread(target=get)
                    thread.start()
                    time.sleep(0.5)
                    while thread.is_alive():
                        QApplication.processEvents()
                        time.sleep(0.02)
                        
                    if not safe_delete:
                        t['last_status'] = 'submitted'
                        t['last_change_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                        self.to_log('Could not fetch results.')
                        self.submitb.setEnabled(True)
                        self.submitb.setText('Submit to NeuroCAAS')
                        return
                    self.to_log('Done fetching results to {0}'.format(localpath))

                    self.aws_view.aws_transfer_queue[i]['last_status'] = 'got_results'

                    if self.delete_results:
                        for f in resultsfiles:
                            self.to_log('Remote delete: {0}'.format(f))
                            f = f.replace(t['awsbucket']+'/','')
                            try:
                                self.aws_view.s3.Object(t['awsbucket'],f).delete()
                            except: # delete already happened?
                                pass
                    if self.delete_inputs:
                        # need to delete the remote data
                        self.to_log('Remote delete: {0}'.format(t['awsdestination']))
                        configpath = os.path.dirname(t['awsdestination'][0])+'/'+'config.yaml' 
                        submitpath = t['awssubmit']#os.path.dirname(t['awsdestination'])+'/'+'submit.json'
                        
                        for a in t['awsdestination']:
                            try:
                                self.aws_view.s3.Object(t['awsbucket'],a).delete()
                            except:
                                pass
                            try:
                                self.aws_view.s3.Object(t['awsbucket'],configpath).delete()
                                self.aws_view.s3.Object(t['awsbucket'],submitpath).delete()
                            except:
                                pass

                    self.to_log('COMPLETED {0}'.format(t['name']))
                
                    if 'analysis_selection' in t['config'].keys():
                        if not 'locanmf' in t['awsbucket']:
                            if 'locaNMF' in t['config']['analysis_selection']:
                                print('Got results, submitting locaNMF analysis')
                                nt = dict(**t)
                                nt['name'] = nt['name'] + '_locanmf'
                                awsfolder = os.path.dirname(nt['awsdestination'][0])
                                foldername = awsfolder.strip('/').split('/')[-1]
                                if not 'locanmf_config' in t.keys():
                                    print('''
There was no locanmf config in the submittion job. 

This happens when you re-submit. You need to resubmit from uploaded data.''')
                                    break
                                nt['config'] = t['locanmf_config']
                                nt['submit'] = t['locanmf_submit']
                                nt['awsbucket'] = NMF_BUCKET 
                                nt['awssubmit'] = ('{0}/'+foldername+'{1}submit.json').format(
                                    nt['submit']['userfolder'],'/')
                                print(nt['awssubmit'])
                                # Which files: labels.json,atlas.npy,brainmask.npy,SVTcorr.npy,U.npy
                                #t['awsdestination'][i] = f.format(t['submit']['userfolder'],'/inputs/')
                                # Check for the landmarks file in the original folder
                                localfolder = os.path.dirname(nt['localpath'][0])
                                landmarksfile = glob(pjoin(localfolder,'*_landmarks.json'))
                                if len(landmarksfile):
                                    landmarksfile = landmarksfile[0]
                                else:
                                    landmarksfile = None
                                if landmarksfile is None:
                                    print('Need to do the alignment first using ncaas open_raw')
                                for d in t['downloaded']:
                                    if 'reduced_spatial' in d:
                                        print('Found reduced spatial.')
                                        U = np.load(d)
                                    if 'config.yaml' in d:
                                        with open(d,'r') as fd:
                                            import yaml
                                            config = yaml.load(fd)
                                            dims = [config['fov_height'],config['fov_width']]
                                
                                if 'dims' in dir() and 'U' in dir() and not landmarksfile is None:
                                    print('Creating the atlas.')
                                    from .allen import atlas_from_landmarks_file, load_allen_landmarks
                                    atlas, areanames, brain_mask = atlas_from_landmarks_file(landmarksfile,
                                                                                             dims = dims,
                                                                                             do_transform = False)
                                    nt['localpath'] = []
                                    nt['awsdestination'] = []
                                    U = U.reshape([*dims,-1])
                                    print('Warping the U matrix.')
                                    from .utils import runpar, im_apply_transform
                                    U[:,0,:] = 0
                                    U[0,:,:] = 0
                                    U[-1,:,:] = 0
                                    U[:,-1,:] = 0
                                    lmarks = load_allen_landmarks(landmarksfile)
                                    U = np.stack(runpar(im_apply_transform,
                                                        U.transpose([2,0,1]),
                                                        M = lmarks['transform'])).transpose([1,2,0])
                                    # spatial
                                    fname = pjoin(localfolder,'results','U_atlas.npy')
                                    np.save(fname,U)
                                    nt['submit']['spatial_data_filename'] = os.path.basename(fname)
                                    f = '{0}/' + foldername + '{1}'+'U_atlas.npy'
                                    nt['localpath'].append(fname)
                                    nt['awsdestination'].append(f.format(nt['submit']['userfolder'],'/inputs/'))
                                    # temporal
                                    if os.path.isfile(pjoin(localfolder,'results','SVTcorr.npy')):
                                        fname = pjoin(localfolder,'results','SVTcorr.npy')
                                    elif os.path.isfile(pjoin(localfolder,'results','reduced_temporal.npy')):
                                        fname = pjoin(localfolder,'results','reduced_temporal.npy')
                                        print('Using reduced temporal for locaNMF temporal!!')
                                    else:
                                        print('No temporal components?')
                                        break
                                    nt['submit']['temporal_data_filename'] = os.path.basename(fname)
                                    f = '{0}/' + foldername + '{1}'+os.path.basename(fname)
                                    nt['localpath'].append(fname)
                                    nt['awsdestination'].append(f.format(nt['submit']['userfolder'],'/inputs/'))

                                    # brainmask
                                    fname = pjoin(localfolder,'results','labels.json')
                                    areanames = [dict(acronym = t[1],
                                                      label=t[0]) for t in areanames]
                                    with open(fname,'w+') as fd:
                                        json.dump(areanames, fd, indent=4)
                                    nt['submit']['areanames_filename'] = os.path.basename(fname)
                                    f = '{0}/' + foldername + '{1}'+'labels.json'
                                    nt['localpath'].append(fname)
                                    nt['awsdestination'].append(f.format(nt['submit']['userfolder'],'/inputs/'))
                                    
                                    # brainmask
                                    fname = pjoin(localfolder,'results','brainmask.npy')
                                    np.save(fname,brain_mask)
                                    nt['submit']['brainmask_filename'] = os.path.basename(fname)
                                    f = '{0}/' + foldername + '{1}'+'brainmask.npy'
                                    nt['localpath'].append(fname)
                                    nt['awsdestination'].append(f.format(nt['submit']['userfolder'],'/inputs/'))

                                    #atlas
                                    fname = pjoin(localfolder,'results','atlas.npy')
                                    np.save(fname,atlas)
                                    nt['submit']['atlas_filename'] = os.path.basename(fname)
                                    f = '{0}/' + foldername + '{1}'+'atlas.npy'
                                    nt['localpath'].append(fname)
                                    nt['awsdestination'].append(f.format(nt['submit']['userfolder'],'/inputs/'))
                                    
                                    nt['last_status'] = 'pending_transfer'
                                    nt['last_change_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

                                    self.aws_view.aws_transfer_queue.append(nt)
                                    #print(nt)
                                    self.refresh_queuelist()
                                    while self.uploading:
                                        QApplication.processEvents()
                                        time.sleep(0.02)
                                    self.process_aws_transfer()
                                    self.aws_view.aws_transfer_queue[-1]['last_status'] = 'submitted'
                                    self.aws_view.aws_transfer_queue[-1]['last_change_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    self.fetching_results = False
                    #self.remove_from_queue(self.queuelist.item(i))
                    
                    self.submitb.setEnabled(True)
                    self.submitb.setText('Submit to NeuroCAAS')
                    return # because we removed an item from the queue, restart the loop
                    

    def remove_from_queue(self,item):
        if not self.uploading:
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
            self.refresh_queuelist()
        else:
            print('Wait until the upload finishes to remove from queue.')
            return
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
            if not 'last_change_time' in itt.keys():
                itt['last_change_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            item.setText('Dataset {0} - state {1} [{2}]'.format(
                itt['name'],
                itt['last_status'].replace('_',' '),
                itt['last_change_time']))
            if itt['last_status'] == 'pending_transfer':
                item.setForeground(QColor(176,107,12))
            elif itt['last_status'] in ['in_transfer','fetching_results']:
                item.setForeground(QColor(7,106,139))
            elif itt['last_status'] in ['uploaded']:
                item.setForeground(QColor(72,33,17))
            elif itt['last_status'] == 'submitted':
                item.setForeground(QColor(22,66,34))
            else:
                item.setForeground(QColor(0,0,0))
                

    def process_aws_transfer(self):
        if self.uploading:
            print('Upload in progress')
            return
        self.uploading = True
        self.submitb.setEnabled(False)
        self.submitb.setText('Transfer in progress')                        
        QApplication.processEvents()
        for i,t in enumerate(self.aws_view.aws_transfer_queue):
            if t['last_status'] == 'pending_transfer':
                # Check if files need to be zipped.
                if t['awsbucket'] == PMD_BUCKET:
                    docompress = [True]
                    from .utils import zipfiles
                    dname = os.path.dirname(t['localpath'][0])
                    fname = t['name']
                    fname = pjoin(dname, fname + '.zip')
                    if os.path.isfile(fname):
                        dlg = QDialog()
                        dlg.setWindowTitle('Found a zip file, do you upload that file?')
                        but = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                        def ok():
                            docompress[0] = False
                            dlg.accept()
                        but.accepted.connect(ok)
                        but.rejected.connect(dlg.accept)
                        l = QVBoxLayout()
                        lab = QLabel('Found a zip file, do you upload that file?')
                        lab.setStyleSheet("font: bold")
                        l.addWidget(lab)
                        l.addWidget(but)
                        dlg.setLayout(l)
                        dlg.exec_()
                    if docompress[0]:
                        #import ipdb
                        #ipdb.set_trace()
                        self.to_log('Zipping {0}'.format(fname))
                        self.submitb.setText('Compressing files')                        
                        QApplication.processEvents()
                        zipfiles(t['localpath'], fname)
                    localpath = [fname]
                    awsdestination = [os.path.dirname(t['awsdestination'][0]) +
                                      '/'+ os.path.basename(fname)]
                else:
                    localpath = t['localpath']
                    awsdestination = t['awsdestination']
                self.submitb.setText('Transfer in progress')                        
                QApplication.processEvents()
                for it, fname in enumerate(localpath):
                    if os.path.isfile(fname):
                        self.pbar.setValue(0)
                        self.pbar.setMaximum(100)
                        self.aws_view.aws_transfer_queue[i]['last_status'] = 'in_transfer'
                        self.aws_view.aws_transfer_queue[i]['last_change_status'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                        self.refresh_queuelist()
                        self.count = 0
                        # check if the file is already there
                        dotransfer = [True]
                        for remotef in self.aws_view.awsfiles:
                            if awsdestination[it] in remotef:
                                dotransfer[0] = False
                        if not dotransfer[0]:
                            dlg = QDialog()
                            dlg.setWindowTitle('File upload {0}'.format(
                                os.path.basename(awsdestination[it])))
                            but = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
                            def ok():
                                dotransfer[0] = True
                                dlg.accept()
                            but.accepted.connect(ok)
                            but.rejected.connect(dlg.accept)
                            l = QVBoxLayout()
                            lab = QLabel('A similar file is already on the cloud, do you want to replace it?')
                            lab.setStyleSheet("font: bold")
                            l.addWidget(lab)
                            l.addWidget(but)
                            dlg.setLayout(l)
                            dlg.exec_()
                        if not dotransfer[0]:
                            print('Skipped {0}'.format(localpath[it]))
                            continue
                        upload = Upload(bucket = t['awsbucket'],
                                        filepath = localpath[it],
                                        destination = awsdestination[it],
                                        s3 = self.aws_view.s3)
                        upload.start()
                        time.sleep(.1)
                        cnt = 0
                        while (upload.isrunning):
                            QApplication.processEvents()
                            self.pbar.setValue(np.ceil(upload.count*99/upload.fsize))
                            time.sleep(0.033)
                            cnt+= 1
                            if np.mod(cnt,3) == 0:
                                self.submitb.setStyleSheet("color: red")
                            else:
                                self.submitb.setStyleSheet("color: black")
                        self.to_log('Transfering {0} {1}'.format(localpath[it],
                                                                 awsdestination[it]))

                        QApplication.processEvents()
                    else:
                        self.to_log('File not found {localpath}'.format(**t))
                        self.remove_from_queue(self.queuelist.item(i))
                        self.submitb.setStyleSheet("color: black")
                        self.submitb.setEnabled(False)
                        return
                    self.to_log('Done transfering {name}'.format(**t))
                t['awsdestination'] = awsdestination
                self.submitb.setStyleSheet("color: black")
                self.pbar.setValue(0)
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'uploaded'
                self.aws_view.aws_transfer_queue[i]['last_change_status'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                t['last_status'] = 'uploaded'
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
                                   t['awssubmit'])
                self.to_log('Submitted analysis {name} to {awssubmit}'.format(**t))
                self.aws_view.aws_transfer_queue[i]['last_status'] = 'submitted'
                self.aws_view.aws_transfer_queue[i]['last_change_time'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            self.refresh_queuelist()
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
                                toadd = dict(
                                    dicttmp, name = os.path.basename(files[0]),
                                    awsdestination = files,
                                    awsbucket=bucketname,
                                    localpath = [pjoin(localpath,os.path.basename(files[0]))],
                                    last_status = "uploaded",
                                    last_change_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
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
                path, config = self.config)
            # this is where we parse the bucket.
            if len(configselection) == 0:
                print('Nothing selected')
                continue
            selection = configselection['selection']
            if (selection[0]['selected'] or # motion
                selection[1]['selected'] or # compression
                selection[2]['selected']):  # hemodynamics
                bucketname = PMD_BUCKET 
            elif selection[3]['selected']:
                bucketname = NMF_BUCKET 
            else:
                print('No analysis were selected ? ')
                return e.accept()
            print('Running analysis on {0}'.format(bucketname))
            
            for t in transfer:
                print('Placing {0} in the transfer queue.'.format(t['name']))
                t['awsbucket'] = bucketname
                t['config'] = configselection[bucketname]['config']
                t['config']['analysis_selection'] = [s['acronym'] for s in selection if s['selected']]
                t['submit'] = configselection[bucketname]['submit']
                if bucketname == PMD_BUCKET:
                    t['awssubmit'] = t['awssubmit'].format(t['submit']['userfolder']+'/inputs/','/')
                else:
                    t['awssubmit'] = t['awssubmit'].format(t['submit']['userfolder'],'/inputs/')

                for i,f in enumerate(t['awsdestination']):
                    if bucketname == PMD_BUCKET:
                        t['awsdestination'][i] = f.format(t['submit']['userfolder']+'/inputs/','/')
                    else:
                        t['awsdestination'][i] = f.format(t['submit']['userfolder'],'/inputs/')

                # Check if it will run locaNMF
                t['selection'] = [s['acronym'] for s in configselection['selection'] if s['selected']]
                if 'locaNMF' in t['selection']:
                    t['locanmf_config'] = configselection[NMF_BUCKET]['config'] 
                    t['locanmf_submit'] = configselection[NMF_BUCKET]['submit'] 
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
            if to_fetch.strip('/') in a:
                files.append(a)
        if not len(files):
            print('No files listed.')
            e.ignore()
            return
        path = [pjoin(self.folder,os.path.basename(p)) for p in paths][0]
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
            fn = f.replace(bucketname,'').strip('/')
            bucket = self.parent.aws_view.s3.Bucket(bucketname)
            def get():
                t = pjoin(f.replace(to_fetch.strip('/'),str(localpath)))
                if not os.path.isdir(os.path.dirname(t)):
                    if os.makedirs(os.path.dirname(t)):
                        os.makedirs(os.path.dirname(t))
                bucket.download_file(fn,t)
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
                f = f.replace(bucketname,'').strip('/')
                try:
                    self.parent.aws_view.s3.Object(bucketname,f).delete()
                except:
                    # If it was already deleted.
                    pass
                self.parent.to_log('Remote delete: {0}'.format(f))

        self.parent.to_log('FINISHED MANUAL COPY {0}'.format(to_fetch))
        self.setSelectionMode(3)
        e.ignore()
    

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
    from botocore.exceptions import ClientError
    try:
        awskeys = ncaas_read_aws_keys()
        s3_connect()
        wind = NCAASwrapper(folder = folder)
        sys.exit(app.exec_())
    except ClientError:
        print('Could not connect to NeuroCAAS, check credentials.')
        cred = CredentialsManager()
        app.exec_()
        awskeys = ncaas_read_aws_keys()
        s3_connect()
        wind = NCAASwrapper(folder = folder)
        sys.exit(app.exec_())
    
