from .utils import *
from PyQt5.QtWidgets import (QApplication,
                             QWidget,
                             QDockWidget,
                             QFormLayout,
                             QHBoxLayout,
                             QGridLayout,
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
pg.setConfigOption('crashWarning', False)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(imageAxisOrder='row-major')
axiscolor = 'k'
class ImageWidget(QWidget):
    def __init__(self):
        super(ImageWidget,self).__init__()

        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.win = pg.GraphicsLayoutWidget()
        self.pl = self.win.addPlot()
        self.pl.getViewBox().invertY(True)
        self.pl.getViewBox().setAspectLocked(True)
        
        self.im = pg.ImageItem()
        self.win.setCentralWidget(self.pl)
        self.pl.addItem(self.im)
        
        self.pl.getAxis('bottom').setPen(axiscolor)
        self.pl.getAxis('left').setPen(axiscolor)
        self.layout.addRow(self.win)

    def _add_hist(self):
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.im)
        self.pl.addItem(self.hist)

class DisplayWidget(ImageWidget):
    def __init__(self, stack,parent=None):
        super(DisplayWidget,self).__init__()
        self.parent = parent
        self.stack = stack
        self.iframe = 0
        tmp = self.stack[:np.clip(100,0,len(self.stack))]
        self.levels = np.percentile(tmp,[5,99])
        self._init_ui()
        self._add_hist()
        self.hist.setHistogramRange(*self.levels)
        self.set_image(0)
        self.win.scene().sigMouseClicked.connect(self.mouseMoved)    

    def _init_ui(self):
        widget = QWidget()
        slayout = QFormLayout()
        widget.setLayout(slayout)
        self.wframe = QSlider(Qt.Horizontal)
        self.wframe.setValue(0)
        self.wframe.setMaximum(len(self.stack)-1)
        self.wframe.setMinimum(0)
        self.wframe.setSingleStep(1)
    
        self.wframelabel = QLabel('Frame {0:d}:'.format(self.wframe.value()))
        slayout.addRow(self.wframelabel, self.wframe)
        self.layout.addRow(widget)
        def uframe(val):
            i = self.wframe.value()
            self.wframelabel.setText('Frame {0:d}:'.format(i))
            self.set_image(i)
            self.parent.roiwidget.p1.setRange(xRange=self.parent.roiwidget.xlim+i)
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
        return t

    def mouseMoved(self,pos):
        modifiers = QApplication.keyboardModifiers()
        pos = self.im.mapFromScene(pos.scenePos())
        #pos = pos.pos()
        if bool(modifiers == Qt.ControlModifier):
            self.parent.roiwidget.add_roi((pos.x(),pos.y()))


class SVDViewer(QMainWindow):
    def __init__(self,stack):
        super(SVDViewer,self).__init__()
        self.setWindowTitle('wfield') 
        self.stack = stack
        self.setDockOptions(QMainWindow.AllowTabbedDocks |
                            QMainWindow.AllowNestedDocks |
                            QMainWindow.AnimatedDocks)

        self.displaywidget = DisplayWidget( self.stack,parent = self)
        self.roiwidget = ROIPlotWidget(stack,self.displaywidget.pl,self.displaywidget.im)
        self.localcorrwidget = LocalCorrelationWidget(stack)

        self.svdtab = QDockWidget('Reconstructed')
        self.svdtab.setWidget(self.displaywidget)
        self.addDockWidget(Qt.RightDockWidgetArea,self.svdtab)
        self.roitab = QDockWidget("ROI", self)
        self.roitab.setWidget(self.roiwidget)
        self.addDockWidget(Qt.BottomDockWidgetArea,self.roitab)
        self.set_dock(self.roitab,False)
        self.lcorrtab = QDockWidget('Pixel correlation')
        self.lcorrtab.setWidget(self.localcorrwidget)
        self.addDockWidget(Qt.RightDockWidgetArea,self.lcorrtab)
        self.tabifyDockWidget(self.lcorrtab,self.svdtab)
        self.show()

    def set_dock(self,dock,floating=False):
        dock.setAllowedAreas(Qt.LeftDockWidgetArea |
                             Qt.RightDockWidgetArea |
                             Qt.BottomDockWidgetArea |
                             Qt.TopDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)
        dock.setFloating(floating)

class LocalCorrelationWidget(ImageWidget):
    def __init__(self, stack,parent=None):
        super(LocalCorrelationWidget,self).__init__()
        self.parent = parent
        self.stack = stack
        U = stack.U
        SVT = stack.SVT
        from .utils_svd import svd_pix_correlation
        self.localcorr = svd_pix_correlation(U,SVT,
                                             dims = self.stack.shape[1:],
                                             norm_svt=True)
        self.levels = [0,1.]
        
        pos = np.array([0, 0.5, 1.])
        color = np.array([[0,0,255,255], [255,255,255,255], [255,0,0,255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0, 1.0, 256)

        self._add_hist()
        self.hist.gradient.setColorMap(cmap)
        self.hist.setHistogramRange(*self.levels)

        self.set_image([0,0])
        self.win.scene().sigMouseMoved.connect(self.mouseMoved)
        
    def set_image(self,xy=[0,0]):
        img = (self.localcorr.get(*xy)+1)/2.
        self.im.setImage(img,levels = self.levels)

    def mouseMoved(self,pos):
        modifiers = QApplication.keyboardModifiers()
        pos = self.im.mapFromScene(pos)
        if bool(modifiers == Qt.ControlModifier):
            self.set_image((int(pos.y()),int(pos.x())))

class ROIPlotWidget(QWidget):
    colors = ['#d62728',
          '#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22']
    penwidth = 1
    def __init__(self,stack, roi_target= None,view=None,npoints = 500):
        super(ROIPlotWidget,self).__init__()	
        layout = QGridLayout()
        self.stack = stack
        self.setLayout(layout)
        self.view = view
        self.roi_target = roi_target
        win = pg.GraphicsLayoutWidget(parent=self)
        self.p1 = win.addPlot()
        self.p1.getAxis('bottom').setPen(axiscolor) 
        self.p1.getAxis('left').setPen(axiscolor) 
        layout.addWidget(win,0,0)
        self.xlim = np.array([0,1000])
        self.N = npoints
        self.rois = []
        self.plots = []
        self.buffers = []
        self.time = np.arange(self.stack.shape[0],dtype='float32')
        self.offset = 0.1
    def add_roi(self,pos):
        pencolor = self.colors[
            np.mod(len(self.plots),len(self.colors))]
        self.rois.append(pg.RectROI(pos=pos,
                                    size=20,
                                    pen=pencolor))
        self.plots.append(pg.PlotCurveItem(pen=pg.mkPen(
            color=pencolor,width=self.penwidth)))
        self.p1.addItem(self.plots[-1])
        self.roi_target.addItem(self.rois[-1])
        self.rois[-1].sigRegionChanged.connect(partial(self.update,i=len(self.rois)-1))

    def items(self):
        return self.rois
    def closeEvent(self,ev):
        for roi in self.rois:
            self.roi_target.removeItem(roi)
        ev.accept()
    def update(self,i):
        X = np.zeros(self.stack.shape[1:],dtype='uint8')
        r = self.rois[i].getArraySlice(X, self.view,axes=(0,1))
        X[r[0][0],r[0][1]]=1
        idx = np.ravel_multi_index(np.where(X==1),self.stack.shape[1:])
        t = np.mean(np.dot(self.stack.U[idx,:].astype('float32'),
                           self.stack.SVT.astype('float32')),axis=0).astype('float32')
        self.plots[i].setData(x = self.time,
                              y = t+self.offset*i)



        #from matplotlib import cm
        # Get the colormap
                
        

            #self.roi[0].plots[0].setData(self.x, self.get_xy(pos.x(),pos.y()), 
            #                 pen = "w", clear = True)
            #self.parent.roiwidget.add_roi(pos)
            #elif


        #win.scene().sigMouseMoved.connect(mouseMoved)    

        #self.x = np.arange(len(self.stack))
        #win2 = pg.GraphicsLayoutWidget()
        #self.plt = win2.addPlot()
        #self.roi = self.plt.plot(x = np.arange(len(self.stack)),
        #                         y = self.get_xy(250,250),color='w')
        #self.plt.addItem(self.roi)
        '''
        from matplotlib.pyplot import get_cmap

        try:
            cmap_plt = get_cmap(cmap)
        except:
            cmap_plt = get_cmap('jet')
        steps = np.linspace(0, 255, 256)
        clrmap_pg = pg.ColorMap(steps, cmap_plt(steps))
        self.im.setLookupTable(clrmap_pg.getLookupTable())
        '''

