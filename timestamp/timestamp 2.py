from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from pyqtgraph.Point import Point
from os.path import expanduser
import os
import math


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


class FolderSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
        self.button=QPushButton('Select Folder')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.folder = ''
        
    def buttonclicked(self):
        prompt = 'testing folderSelector'
        self.folder = QFileDialog.getExistingDirectory(g.m, "Select recording folder.", expanduser("~"), QFileDialog.ShowDirsOnly)
        self.label.setText('...'+os.path.split(self.folder)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])    



class Timestamp(BaseProcess_noPriorWindow):
    """
    Timestamp
    
    
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)

    def __call__(self):
        '''
        
        '''
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)


    def gui(self):
        self.gui_reset()
        framerate=pg.SpinBox()
        
        if hasattr(g.win,'framerate'):
            framerate.setValue(g.win.framerate)
        elif 'framerate' in g.settings.d.keys():
            framerate.setValue(g.settings['framerate'])
        
        framerate.setRange(0,1000000)
        framerate.setDecimals(10)
        
        show = CheckBox(); show.setChecked(True)
        
        #windows
        self.window_button = WindowSelector()
        
        #self.exportFolder = FolderSelector('*.txt')
        self.items.append({'name': 'window', 'string': 'Select Window', 'object': self.window_button})               
        self.items.append({'name':'framerate','string':'Frame Rate (Hz)','object':framerate})
        self.items.append({'name':'show','string':'Show','object':show})
        super().gui()



    def addIndex(self,framerate,show=True,keepSourceWindow=None):
        w = self.getValue('window')
        
        if show:
            w.framerate=framerate
            g.settings['framerate']=framerate
            if hasattr(w,'timeStampLabel') and w.timeStampLabel is not None:
                return
            w.timeStampLabel= pg.TextItem(html="<span style='font-size: 12pt;color:white;background-color:None;'>0 ms</span>")
            w.imageview.view.addItem(w.timeStampLabel)
            w.sigTimeChanged.connect(w.updateTimeStampLabel)
        else:
            if hasattr(w,'timeStampLabel') and w.timeStampLabel is not None:
                w.imageview.view.removeItem(w.timeStampLabel)
                w.timeStampLabel=None
                w.sigTimeChanged.disconnect(w.updateTimeStampLabel)
        return None


    def preview(self):
        framerate=self.getValue('framerate')
        show=self.getValue('show')
        self.addIndex(framerate,show)
               

 
timestamp = Timestamp()
