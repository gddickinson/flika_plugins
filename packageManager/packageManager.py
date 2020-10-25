#from __future__ import (absolute_import, division,print_function, unicode_literals)
#from future.builtins import (bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
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
import json
import os

#import conda.cli.python_api as Conda

from .defaultPackageList import defaultCondaPackageList

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
        self.condaENV = (os.environ['CONDA_DEFAULT_ENV'])
        return

    def __call__(self):
        pass
        return       

    def gui(self):
        self.gui_reset()
        
        self.showPackages_button = QtWidgets.QPushButton('Show Packages')
        self.showPackages_button.pressed.connect(self.showPackages)   
        
        self.installPackages_button = QtWidgets.QPushButton('Install Packages')
        self.installPackages_button.pressed.connect(self.installPackages)         
        
        self.items.append({'name':'getPackages','string':'Show currently installed python packages: ', 'object': self.showPackages_button})  
        self.items.append({'name':'installPackages','string':'Install default python packages: ', 'object': self.installPackages_button})  
        
        super().gui()
        return

    def showPackages(self):
        proc = subprocess.run(["conda", "list", "--json"], text=True, capture_output=True)
        self.packageList = json.loads(proc.stdout)
        self.listDisplay.clear()
        for item in self.packageList:
            self.listDisplay.addItem(item)
        self.listDisplay.show()
        return


    def installPackages(self, default=True, level='>='):
        if default:
            installPackages = []
            for package in defaultCondaPackageList:
                item = str(package['name'] + level + package['version'])
                installPackages.append(item)
            #print(installPackages)
        packageStr = " ".join(installPackages)
        
        self.install(installPackages, level)

        

    def install(self, packages,level):
        condaInstall = []
        pipInstall = []
        uninstalled = []
        for package in packages:
            print('-------------------')            
            print(package)
            print('-------------------')
            try:
                try:
                    try:
                        subprocess.check_call([sys.executable, '-m', 'conda', 'install', 
                                       package, '-y'])
                        condaInstall.append(package)
               
                    except:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                               package, '--no-cache-dir'])
                        pipInstall.append(package)
            
                except:
                    package = package.split(level)[0]
                    try:
                        subprocess.check_call([sys.executable, '-m', 'conda', 'install', 
                                       package, '-y'])
                        condaInstall.append(package)
               
                    except:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                               package, '--no-cache-dir'])
                        pipInstall.append(package)                
            except:
               uninstalled.append(package) 
        
        print('Conda examined: ', condaInstall)
        print('-------------------')
        print('Pip examined: ', pipInstall)        
        print('-------------------') 
        print('Uninstalled: ', uninstalled)          
        print('Restart FLIKA to load new libraries')
        print('-------------------')
        return        


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