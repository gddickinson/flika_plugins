from __future__ import (absolute_import, division,print_function, unicode_literals)
from future.builtins import (bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from qtpy import QtWidgets, QtCore, QtGui
import flika
flika_version = flika.__version__
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
import time
from distutils.version import StrictVersion
import sys
import subprocess

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow

#Get OpenCV version
#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#print("OpenCV version %s.%s.%s" % (major_ver, minor_ver, subminor_ver))



class PackageManager(BaseProcess_noPriorWindow):
    """
    Flika plugin for managing python packages
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.packageList = []
        self.listDisplay = ListWindow()
        return

    def __call__(self):
        pass
        return       

    def gui(self):
        self.gui_reset()
        
        self.getPackages = QtWidgets.QPushButton('Get Packages')
        self.getPackages.pressed.connect(self.showPackages)        
        
        self.items.append({'name':'getPackages','string':'Show currently installed python packages: ', 'object': self.getPackages})  
        super().gui()
        return

    def showPackages(self):
        import json
        proc = subprocess.run(["conda", "list", "--json"], text=True, capture_output=True)
        self.packageList = json.loads(proc.stdout)
        self.listDisplay.clear()
        for item in self.packageList:
            self.listDisplay.addItem(item)
        self.listDisplay.show()
        return


    def conda_install(self, environment, *package):
        proc = run(["conda", "install", "--quiet"] + packages,
                   text=True, capture_output=True)
        return json.loads(proc.stdout)

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
        #self.ui.close()
        #event.accept()
        return


class ListWindow(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        self.listwidget = QtWidgets.QListWidget()
        self.listwidget.clicked.connect(self.clicked)
        layout.addWidget(self.listwidget)

    def addItem(self, item):
        n = self.listwidget.count()
        self.listwidget.insertItem(n, str(item['name']+' -- '+item['version']))
        return

    def clicked(self, qmodelindex):
        item = self.listwidget.currentItem()
        print(item)
        
    def clear(self):
        self.listwidget.clear()

packageManager = PackageManager()