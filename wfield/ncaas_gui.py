from .utils import *
import sys
from PyQt5.QtWidgets import (QApplication,
                             QWidget,
                             QMainWindow,
                             QDockWidget,
                             QFormLayout,
                             QHBoxLayout,
                             QVBoxLayout,
                             QGridLayout,
                             QTreeWidgetItem,
                             QTreeView,
                             QTextEdit,
                             QCheckBox,
                             QLabel,
                             QFileSystemModel,
                             QAbstractItemView)

from PyQt5 import QtCore
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt, QTimer,QMimeData

import json
tempfile = pjoin(os.path.expanduser('~'),'.wfield','tempfile')

defaultconfig = dict(analysis = 'cshl-wfield-preprocessing',
                     userfolder = 'ChurchlandLab',
                     instance_type =  'r5.16xlarge',
                     config = dict(block_height = 90,
                                   block_width = 90,
                                   frame_rate = 30,
                                   max_components = 15,
                                   num_sims = 64,
                                   overlapping = True,
                                   window_length = 7200))

# Function to add credentials
def ncaas_set_aws_keys(ncaas_login,add_default_region = True):
    fname = pjoin(os.path.expanduser('~'),'.aws','credentials')
    cred = '''[default]
aws_access_key_id = {Access Key}
aws_secret_access_key = {Secret Access Key}
'''
    dirname = os.path.dirname(fname)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)    
    with open(fname,'w') as fd:
        fd.write(cred.format(**ncaas_login))
    fname = pjoin(os.path.expanduser('~'),'.aws','config')
    if add_default_region:
        conf = '''[default]
region=us-east-1'''
        with open(fname,'w') as fd:
            fd.write(conf)

try:
    import boto3
except:
    print('boto3 not installed, do...instructions ')

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

class TextEditor(QDockWidget):
    def __init__(self,path,s3=None,bucket=None,refresh_interval = 2):
        super(TextEditor,self).__init__()
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
        ckbox.stateChanged.connect(watch)
        lay.addWidget(w)
        
    def refresh_original(self):
        if not self.s3 is None:
            bucket = self.s3.Bucket(self.bucketname)
            bucket.download_file(self.path,tempfile)
            with open(tempfile,'r') as f:
                return f.read()

class NCAASwrapper(QMainWindow):
    def __init__(self,folder = '.', config = pjoin(os.path.expanduser('~'),
                                                   '.wfield','ncaas_config.json')):
        super(NCAASwrapper,self).__init__()

        folder = os.path.abspath(folder)
        if not config is dict: # then it is a filepath
            if not os.path.exists(os.path.dirname(config)):
                os.makedirs(os.path.dirname(config))
            if not os.path.exists(config):
                with open(config,'w') as f:
                    print('Creating config from defaults [{0}]'.format(config))
                    json.dump(defaultconfig,f,indent = 4)
            with open(config,'r') as f:
                config = json.load(f)

        self.config = config
        self.folder = folder
        mainw = QWidget()
        self.setCentralWidget(mainw)
        lay = QHBoxLayout()
        mainw.setLayout(lay)
        # Filesystem browser
        self.fs_view = FilesystemView(folder)
        # Add the widget with label
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        l.addWidget(QLabel('<b>' + folder + '<\b>'))
        l.addWidget(self.fs_view)
        lay.addWidget(w)
        
        # AWS browser
        self.aws_view = AWSView(config,parent=self)


        #    print(f.key)
        # Add the widget with label
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        bucketname = '/'.join([self.config['analysis'],
                               self.config['userfolder']])
        l.addWidget(QLabel('<b>' + 'NeuroCAAS - {0}'.format(bucketname) + '<\b>'))
        l.addWidget(self.aws_view)
        lay.addWidget(w)
        
        #import ipdb
        #ipdb.set_trace()
        
        self.show()

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

        self.config = config
        self.s3 = s3_connect()
        self.bucketname = self.config['analysis']
        self.awsfiles = []


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
        e.ignore() # Dont drop the remote table
        
class AWSItemModel(QStandardItemModel):
    def __init__(self):
        super(AWSItemModel,self).__init__()

    def mimeData(self,idx):
        tt = QMimeData()
        tt.setText(','.join(get_tree_path(idx)))
        return tt
        
class FilesystemView(QTreeView):
    def __init__(self,folder):
        super(FilesystemView,self).__init__()
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
        print(paths)
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
    wind = NCAASwrapper(folder = '.')
    sys.exit(app.exec_())
