from .utils import *
from PyQt5.QtWidgets import (QApplication,
                             QWidget,
                             QDockWidget,
                             QFormLayout,
                             QHBoxLayout,
                             QMainWindow,
                             QGroupBox,
                             QSlider,
                             QWidgetAction,
                             QListWidget,
                             QComboBox,
                             QPushButton,
                             QLabel,
                             QTextEdit,
                             QDoubleSpinBox,
                             QSpinBox,
                             QCheckBox,
                             QProgressBar,
                             QFileDialog,
                             QAction)

from PyQt5 import QtCore
from PyQt5.QtCore import Qt,QTimer

import pyqtgraph as pg
pg.setConfigOption('background', [0,0,0])
pg.setConfigOption('crashWarning', True)
pg.setConfigOptions(imageAxisOrder='row-major')


class DisplayWidget(QWidget):
    def __init__(self, stack,parent=None):
        super(DisplayWidget,self).__init__()
        self.parent = parent
        self.stack = stack
        self.iframe = 0
        self.levels = [np.min(self.stack[0]),2*np.max(self.stack[0])]
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        self.setLayout(layout)
        win = pg.GraphicsLayoutWidget()
        self.pl = win.addPlot()
        self.pl.getViewBox().invertY(True)
        self.pl.getViewBox().setAspectLocked(True)
        
        self.im = pg.ImageItem()
        self.im.setImage(self.stack[0])
        self.pl.addItem(self.im)
        self.pl.getAxis('bottom').setPen('w')
        self.pl.getAxis('left').setPen('w')
        
        def mouseMoved(pos):
            modifiers = QApplication.keyboardModifiers()
            if bool(modifiers == Qt.ControlModifier):
                pos = self.im.mapFromScene(pos)
                
                self.roi.setData(self.x, self.get_xy(pos.x(),pos.y()), 
                                 pen = "w", clear = True)


        win.scene().sigMouseMoved.connect(mouseMoved)    

        self.x = np.arange(len(self.stack))
        win2 = pg.GraphicsLayoutWidget()
        self.plt = win2.addPlot()
        self.roi = self.plt.plot(x = np.arange(len(self.stack)),
                                 y = self.get_xy(250,250),color='w')
        self.plt.addItem(self.roi)
        
        widget = QWidget()
        slayout = QFormLayout()
        widget.setLayout(slayout)
        layout.addRow(widget)
        self.wframe = QSlider(Qt.Horizontal)
        self.wframe.setValue(0)
        self.wframe.setMaximum(len(self.stack)-1)
        self.wframe.setMinimum(0)
        self.wframe.setSingleStep(1)
    
        self.wframelabel = QLabel('Frame {0:d}:'.format(self.wframe.value()))
        slayout.addRow(self.wframelabel, self.wframe)
        layout.addRow(win)
        layout.addRow(widget)
        layout.addRow(win2)
        def uframe(val):
            i = self.wframe.value()
            self.wframelabel.setText('Frame {0:d}:'.format(i))
            self.set_image(i)
        self.wframe.valueChanged.connect(uframe)
        
    def set_image(self,i=None):
        if not i is None:
            self.iframe = i
        img = self.stack[self.iframe]
        self.im.setImage(img,levels = self.levels)
    def get_xy(self,x,y):
        x = int(np.clip(x,0,self.stack.shape[1]))
        y = int(np.clip(y,0,self.stack.shape[2]))
        idx = np.ravel_multi_index((x,y),self.stack.shape[1:])
        t = np.dot(self.stack.U[idx,:],self.stack.SVT)
        return t - np.percentile(t,10)

class SVDViewer(QMainWindow):
    def __init__(self,stack):
        super(SVDViewer,self).__init__()
        self.stack = stack
        self.setDockOptions(QMainWindow.AllowTabbedDocks |
                            QMainWindow.AllowNestedDocks |
                            QMainWindow.AnimatedDocks)
        self.displaywidget = DisplayWidget(self.stack)
        dock = QDockWidget('SVD display')
        dock.setWidget(self.displaywidget)
        self.addDockWidget(Qt.RightDockWidgetArea,dock)
        self.show()
