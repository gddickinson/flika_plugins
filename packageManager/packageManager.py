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

        self.installSinglePackage_button = QtWidgets.QPushButton('Start Dialog')
        self.installSinglePackage_button.pressed.connect(self.installPackage)         
        
        self.items.append({'name':'getPackages','string':'Show currently installed python packages: ', 'object': self.showPackages_button})  
        self.items.append({'name':'installPackages','string':'Install default python packages: ', 'object': self.installPackages_button})  
        self.items.append({'name':'installPackage','string':'Install single python package: ', 'object': self.installSinglePackage_button})  
        
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

    def queryDialog(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)

        msg.setText("Package Update Warning")
        msg.setInformativeText("Do you wish to proceed with Python Package update?")
        msg.setWindowTitle("packageManger dialog")
        #msg.setDetailedText("The packages are:", defaultCondaPackageList) #too many packages to display
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        #msg.buttonClicked.connect(msgbtn)
        retval = msg.exec_()        
      
        return retval
        

    def installPackage(self):
        self.dialogWin = getPackageDialog()
        self.dialogWin.show()
        return

    def installPackages(self, default=True, level='>='):

        retval = self.queryDialog()
        #1024 = Ok
        if retval != 1024:
            return
        
        print('Package Update running...')

        
        if default:
            installPackages = []
            for package in defaultCondaPackageList:
                item = str(package['name'] + level + package['version'])
                installPackages.append(item)
            #print(installPackages)
        packageStr = " ".join(installPackages)
        
        self.install(installPackages, level)

        print('Package Update finished!')        

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
        self.dialogWin.close()
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

class getPackageDialog(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(getPackageDialog, self).__init__(parent)
		
        layout = QtWidgets.QFormLayout()
        self.btn = QtWidgets.QPushButton("Choose from list")
        self.btn.clicked.connect(self.getItem)
        self.btn1 = QtWidgets.QPushButton("Type name:")
        #self.btn1.clicked.connect(self.gettext)
		
        self.le = QtWidgets.QLineEdit()
        layout.addRow(self.btn)
        layout.addRow(self.btn1, self.le)
		
        #self.le1 = QtWidgets.QLineEdit()
        #layout.addRow(self.btn1,self.le1)
        self.btn2 = QtWidgets.QPushButton("Enter minimun version:")
        #self.btn2.clicked.connect(self.getmin)
               		
        self.le2 = QtWidgets.QLineEdit()
        layout.addRow(self.btn2,self.le2)

        self.btn3 = QtWidgets.QPushButton("Enter maximum version:")
        #self.btn3.clicked.connect(self.getmax)    

        self.le3 = QtWidgets.QLineEdit()
        layout.addRow(self.btn3,self.le3)

        self.btn4 = QtWidgets.QPushButton("Go")
        self.btn4.clicked.connect(self.go)      
        layout.addRow(self.btn4)

        self.setWindowTitle("Single Package Install Dialog")        
        self.setLayout(layout)

    
		
    def getItem(self):
        defaultPackages = []
        for package in defaultCondaPackageList:
            item = str(package['name'])
            defaultPackages.append(item)
       		
        item, ok = QtWidgets.QInputDialog.getItem(self, "select input dialog", 
                                                  "list of pacakges", defaultPackages, 0, False)
			
        if ok and item:
            self.le.setText(item)
			
    def gettext(self):
        text, ok = QtWidgets.QInputDialog.getText(self, 'Text Input Dialog', 'Enter package name:')
		
        if ok:
            self.le.setText(str(text))
			
    def getmin(self):
        text,ok = QtWidgets.QInputDialog.getText(self,"Text input dialog","enter minimum version")
		
        if ok:
            self.le2.setText(str(text))

    def getmax(self):
        text,ok = QtWidgets.QInputDialog.getText(self,"Text input dialog","enter maximum version")
		
        if ok:
            self.le3.setText(str(text))

    def go(self):
        pkg = self.le.text()
        minVersion = self.le2.text()
        maxVersion = self.le3.text() 

        print('Attempting to install:{}, minimum version = {}, maximum version = {}'.format(pkg,minVersion,maxVersion))
        
        if minVersion == maxVersion or (minVersion != '' and maxVersion ==''):
            item = str(pkg + '==' + minVersion)
        
        elif maxVersion > minVersion:
            item = str(pkg + '>=' + minVersion + ',' + '<' + maxVersion)

        elif minVersion == '' and maxVersion == '':
            item = pkg
        
        elif minVersion == '' and maxVersion != '':
            item = str(pkg + '<=' + maxVersion)
        
        else:
            print('Min/Max values incompatible')   
            
        self.install(item)
            
            
    def install(self, package):
        try:
            try:
                print('Running: ', + ' '+'conda'+' '+ 'install'+ ' '+ package + ' ' + '-y')
                subprocess.check_call([sys.executable, '-m', 'conda', 'install', 
                                       package, '-y'])
                   
            except:
                print('Running: ' + ' '+'pip'+' '+ 'install'+ ' '+ package + ' ' + '--no-cache-dir')
                subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                       package, '--no-cache-dir'])

        except:
            print('Install of {} not performed'.format(package))
