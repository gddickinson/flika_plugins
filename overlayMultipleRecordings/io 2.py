#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:53:24 2023

@author: george
"""

import flika
from flika import global_vars as g
import os
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

# function to prompt user to select a file for opening
def open_file_gui(prompt="Open File", directory=None, filetypes=''):
    """ File dialog for opening an existing file, isolated to handle tuple/string return value

    Args:
        prompt (str): string to display at the top of the window
        directory (str): initial directory to open
        filetypes (str): argument for filtering file types separated by ;; (*.png) or (Images *.png);;(Other *.*)

    Returns:
        str: the file (path+file+extension) selected, or None
    """
    filename = None
    if directory is None:
        filename = g.settings['filename']
        try:
            directory = os.path.dirname(filename)
        except:
            directory = None
    if directory is None or filename is None:
        filename = QFileDialog.getOpenFileName(g.m, prompt, '', filetypes)
    else:
        filename = QFileDialog.getOpenFileName(g.m, prompt, filename, filetypes)
    if isinstance(filename, tuple):
        filename, ext = filename
        if ext and '.' not in filename:
            filename += '.' + ext.rsplit('.')[-1]
    if filename is None or str(filename) == '':
        g.m.statusBar().showMessage('No File Selected')
        return None
    else:
        return str(filename)

class FileSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a file to save.  Once you do, it sets self.filename and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*', mainGUI=None):
        QWidget.__init__(self)

        self.mainGUI = mainGUI

        self.button=QPushButton('Load Data')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.filename = ''
        self.columns = []
        self.pixelSize = self.mainGUI.trackPlotOptions.pixelSize_selector.value()    #nanometers


    def buttonclicked(self):
        if g.win == None:
            g.alert('Load tiff stack and set as current window first')
            return
        prompt = 'testing fileSelector'
        self.filename = open_file_gui(prompt, filetypes=self.filetypes)
        self.label.setText('...'+os.path.split(self.filename)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.filename

    def setValue(self, filename):
        self.filename = str(filename)
        self.label.setText('...' + os.path.split(self.filename)[-1][-20:])


class FileSelector_overlay(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a file to save.  Once you do, it sets self.filename and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)

        self.button=QPushButton('Load Tiff')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.filename = ''

    def buttonclicked(self):
        prompt = 'select tiff to overlay'
        self.filename = open_file_gui(prompt, filetypes=self.filetypes)
        self.label.setText('...'+os.path.split(self.filename)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.filename

    def setValue(self, filename):
        self.filename = str(filename)
        self.label.setText('...' + os.path.split(self.filename)[-1][-20:])
