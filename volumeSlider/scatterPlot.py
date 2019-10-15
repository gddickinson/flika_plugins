from qtpy import QtWidgets
from pyqtgraph.dockarea import *
#from pyqtgraph import mkPen
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
#import copy
#import pyqtgraph.opengl as gl
from OpenGL.GL import *
#from qtpy.QtCore import Signal
from .helperFunctions import *
#########################################################################################
#############            3D Matlibplot scatter plot            ##########################
#########################################################################################
class plot3D_options(QtWidgets.QDialog):
    def __init__(self, viewerInstance, prob, threshold, parent = None):
        super(plot3D_options, self).__init__(parent)

        self.viewer = viewerInstance

        self.prob = prob
        self.threshold = threshold

        #window geometry
        windowGeometry(self, left=300, top=300, height=300, width=200)

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Amount of downsampling (0-1)")
        self.SpinBox1 = QtWidgets.QDoubleSpinBox()
        self.SpinBox1.setDecimals(4)
        self.SpinBox1.setRange(0,1.0000)
        self.SpinBox1.setValue(self.prob)

        self.spinLabel2 = QtWidgets.QLabel("Threshold level")
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,10000)
        self.SpinBox2.setValue(self.threshold)

        #checkboxes
        self.checkBox1_label = QtWidgets.QLabel("Display array outline")
        self.checkBox1 = QtWidgets.QCheckBox()
        self.checkBox2_label = QtWidgets.QLabel("Display plot axis")
        self.checkBox2 = QtWidgets.QCheckBox()
        
        #buttons
        #self.button1 = QtWidgets.QPushButton("Close")

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)

        layout.addWidget(self.spinLabel1, 1, 0)
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.spinLabel2, 2, 0)
        layout.addWidget(self.SpinBox2, 2, 1)
        layout.addWidget(self.checkBox1_label, 3, 0)
        layout.addWidget(self.checkBox1, 3, 1)
        layout.addWidget(self.checkBox2_label, 4, 0)
        layout.addWidget(self.checkBox2, 4, 1)
        #layout.addWidget(self.button1, 5, 2)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("3D Plot Options")

        #connect spinboxes
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.SpinBox2.valueChanged.connect(self.spinBox2ValueChange)

        #connect checkboxes
        self.checkBox1.stateChanged.connect(self.checkBox1ValueChange)
        self.checkBox2.stateChanged.connect(self.checkBox2ValueChange)

        #connect buttons
        #self.button1.clicked.connect(self.close)

        return

    def spinBox1ValueChange(self, value):
        self.viewer.viewer.setProb(value)
        return

    def spinBox2ValueChange(self, value):
        self.viewer.viewer.setThreshold(value)
        return

    def checkBox1ValueChange(self, value):
        self.viewer.viewer.setPlotCube(value)
        return

    def checkBox2ValueChange(self, value):
        self.viewer.viewer.setPlotAxis((not value))
        return

