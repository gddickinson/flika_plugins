#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:59:21 2023

@author: george
"""

import pyqtgraph as pg
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from distutils.version import StrictVersion
import os, shutil, subprocess
import errno

import flika
from flika.window import Window
import flika.global_vars as g

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

from .helperFunctions import *

class Scale_Bar_ROIzoom(BaseProcess):
    ''' scale_bar(width_NoUnits, width_pixels, font_size, color, background, offset, location, show=True)

    Parameters:
        width_NoUnits (float): width
        width_pixels (float): width in pixels
        font_size (int): size of the font
        color (string): ['Black', White']
        background (string): ['Black','White', 'None']
        location (string): ['Lower Right','Lower Left','Top Right','Top Left']
        offset (int): manual positioning of bar and label
        show (bool): controls whether the Scale_bar is displayed or not
    '''

    def __init__(self, roiGUI):
        super().__init__()
        self.roiGUI = roiGUI

    def gui(self):
        self.gui_reset()

        self.w = self.roiGUI.w1
        width_NoUnits=QSpinBox()
        width_NoUnits.setRange(1,10000)

        width_pixels=QDoubleSpinBox()
        width_pixels.setRange(.001,1000000)
        #width_pixels.setRange(1,self.roiGUI.mx)

        font_size=QSpinBox()

        units=ComboBox()
        units.addItem("nm")
        units.addItem("µm")

        color=ComboBox()
        color.addItem("White")
        color.addItem("Black")
        background=ComboBox()
        background.addItem('None')
        background.addItem('Black')
        background.addItem('White')
        location=ComboBox()
        location.addItem('Lower Right')
        location.addItem('Lower Left')
        location.addItem('Top Right')
        location.addItem('Top Left')
        show=CheckBox()

        offset=QSpinBox()

        font_size.setValue(12)
        width_pixels.setValue(1.00)
        width_NoUnits.setValue(108)

        font_size.setValue(0)

        show.setChecked(True)
        self.items.append({'name':'width_NoUnits','string':'Width of bar','object':width_NoUnits})
        self.items.append({'name':'width_unit','string':'Width of bar units','object':units})
        self.items.append({'name':'width_pixels','string':'Width of bar in pixels','object':width_pixels})
        self.items.append({'name':'font_size','string':'Font size','object':font_size})
        self.items.append({'name':'color','string':'Color','object':color})
        self.items.append({'name':'background','string':'Background','object':background})
        self.items.append({'name':'location','string':'Location','object':location})
        self.items.append({'name':'offset','string':'Offset','object':offset})
        self.items.append({'name':'show','string':'Show','object':show})

        super().gui()
        self.preview()

    def __call__(self,width_NoUnits, width_pixels, font_size, color, background,location,offset,show=True,keepSourceWindow=None):

        if show:
            if hasattr(self.roiGUI,'scaleBarLabel') and self.roiGUI.scaleBarLabel is not None:
                self.w.view.removeItem(self.roiGUI.scaleBarLabel.bar)
                self.w.view.removeItem(self.roiGUI.scaleBarLabel)
                self.w.view.sigResized.disconnect(self.updateBar)
            if location=='Top Left':
                anchor=(0,0)
                pos=[0,0]
            elif location=='Top Right':
                anchor=(0,0)
                pos=[self.roiGUI.mx,0]
            elif location=='Lower Right':
                anchor=(0,0)
                pos=[self.roiGUI.mx,self.roiGUI.my]
            elif location=='Lower Left':
                anchor=(0,0)
                pos=[0,self.roiGUI.my]
            self.roiGUI.scaleBarLabel= pg.TextItem(anchor=anchor, html="<span style='font-size: {}pt;color:{};background-color:{};'>{} {}</span>".format(font_size, color, background,width_NoUnits,self.getValue('width_unit')))
            self.roiGUI.scaleBarLabel.setPos(pos[0],pos[1])
            self.roiGUI.scaleBarLabel.flika_properties={item['name']:item['value'] for item in self.items}
            self.w.view.addItem(self.roiGUI.scaleBarLabel)
            if color=='White':
                color255=[255,255,255,255]
            elif color=='Black':
                color255=[0,0,0,255]
            textRect=self.roiGUI.scaleBarLabel.boundingRect()

            if location=='Top Left':
                barPoint=QPoint(int(0), int(textRect.height()))
            elif location=='Top Right':
                barPoint=QPoint(int(-width_pixels), int(textRect.height()))
            elif location=='Lower Right':
                barPoint=QPoint(int(-width_pixels), int(-textRect.height()))
            elif location=='Lower Left':
                barPoint=QPoint(int(0), int(-textRect.height()))

            bar = QGraphicsRectItem(QRectF(barPoint,QSizeF(width_pixels,int(font_size/3))))
            bar.setPen(pg.mkPen(color255)); bar.setBrush(pg.mkBrush(color255))
            self.w.view.addItem(bar)
            #bar.setParentItem(self.roiGUI.scaleBarLabel)
            self.roiGUI.scaleBarLabel.bar=bar
            self.w.view.sigResized.connect(self.updateBar)
            self.updateBar()

        else:
            if hasattr(self.roiGUI,'scaleBarLabel') and self.roiGUI.scaleBarLabel is not None:
                self.w.view.removeItem(self.roiGUI.scaleBarLabel.bar)
                self.w.view.removeItem(self.roiGUI.scaleBarLabel)
                self.roiGUI.scaleBarLabel=None
                self.w.view.sigResized.disconnect(self.updateBar)
        return None

    def updateBar(self):
        width_pixels=self.getValue('width_pixels')
        location=self.getValue('location')
        offset=self.getValue('offset')
        view = self.w.view
        textRect=self.roiGUI.scaleBarLabel.boundingRect()
        textWidth=textRect.width()*view.viewPixelSize()[0]
        textHeight=textRect.height()*view.viewPixelSize()[1]

        if location=='Top Left':
            barPoint=QPoint(int(0) + offset, int(1.3*textHeight))
            self.roiGUI.scaleBarLabel.setPos(QPointF(offset + width_pixels/2-textWidth/2,0))
        elif location=='Top Right':
            barPoint=QPoint(int(self.roiGUI.mx-width_pixels) - offset, int(1.3*textHeight))
            self.roiGUI.scaleBarLabel.setPos(QPointF(self.roiGUI.mx-width_pixels/2-textWidth/2 - offset,0))
        elif location=='Lower Right':
            barPoint=QPoint(int(self.roiGUI.mx-width_pixels) -offset, int(self.roiGUI.my-1.3*textHeight))
            self.roiGUI.scaleBarLabel.setPos(QPointF(int(self.roiGUI.mx-width_pixels/2-textWidth/2)-offset,int(self.roiGUI.my-textHeight)))
        elif location=='Lower Left':
            barPoint=QPoint(int(0) +offset, int(self.roiGUI.my-1.3*textHeight))
            self.roiGUI.scaleBarLabel.setPos(QPointF(QPointF(offset + width_pixels/2-textWidth/2,self.roiGUI.my-textHeight)))
        self.roiGUI.scaleBarLabel.bar.setRect(QRectF(barPoint, QSizeF(width_pixels,textHeight/4)))

    def preview(self):
        width_NoUnits=self.getValue('width_NoUnits')
        width_pixels=self.getValue('width_pixels')
        font_size=self.getValue('font_size')
        color=self.getValue('color')
        background=self.getValue('background')
        location=self.getValue('location')
        offset=self.getValue('offset')
        show=self.getValue('show')
        self.__call__(width_NoUnits, width_pixels, font_size, color, background, location, offset, show)

def extractListElement(l, pos):
    return list(list(zip(*l))[pos])

class CheckableComboBox(QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItemDirect(self,text):
        item = QStandardItem()
        item.setText(text)

        item.setData(text)

        item.setData(Qt.Checked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItem(self, text, data=None, unchecked=True):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        if unchecked:
            item.setData(Qt.Unchecked, Qt.CheckStateRole)
        else:
            item.setData(Qt.Checked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res

    def currentItems(self):
        # Return the list of selected items
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res

    def setItems(self, items):
        """
        *items* may be a list, a tuple, or a dict.
        If a dict is given, then the keys are used to populate the combo box
        and the values will be used for both value() and setValue().
        """
        currentItems = self.currentItems()
        self.clear()
        for item in currentItems:
            self.addItemDirect(item)

        for item in items:
            if item not in self.currentData():
                self.addItem(item)


    def items(self):
        return self.items.copy()

    def value(self):
        """
        If items were given as a list of strings, then return the currently
        selected text. If items were given as a dict, then return the value
        corresponding to the currently selected key. If the combo list is empty,
        return None.
        """
        if self.count() == 0:
            return None

        return self.currentData()

def custom_symbol(symbol: str, font: QFont = QFont("San Serif")):
    """Create custom symbol with font"""
    # We just want one character here, comment out otherwise
    #assert len(symbol) == 1
    pg_symbol = QPainterPath()
    font.setBold(True)
    pg_symbol.addText(0, 0, font, symbol)
    # Scale symbol
    br = pg_symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QTransform()
    tr.scale(scale, scale)
    tr.translate(-br.x() - br.width() / 2., -br.y() - br.height() / 2.)
    return tr.map(pg_symbol)

# def QImageToCvMat(incomingImage):
#     '''  Converts a QImage into an opencv MAT format  '''
#     incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

#     width = incomingImage.width()
#     height = incomingImage.height()

#     ptr = incomingImage.bits()
#     ptr.setsize(height * width * 4)
#     arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
#     return arr


class ROIPLOT():
    """
    A class for displaying ROI image with scrolling update of locs positions and intensity trace.
    """
    def __init__(self, mainGUI):
        super().__init__()

        # Initialize the main GUI instance
        self.mainGUI = mainGUI
        self.dataWindow = None
        self.tracksInView = []
        self.selectedPoints = []
        self.trackToDisplay = None
        self.ROIplotInitiated = False
        self.mx = None
        self.my = None
        self.nFrames = None
        self.scaleBarLabel = None
        self.traceLegend = None
        self.pathitems = []

        self.scale_bar = Scale_Bar_ROIzoom(self)

        # Create a dock window and add a dock
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1400,350)
        self.win.setWindowTitle('ROI Plot')
        self.d1 = Dock("ROI Zoom Window", size=(400, 350))
        self.d2 = Dock("Intensity Trace", size=(500, 350))
        self.d3 = Dock("Options Panel", size=(200, 350))
        self.area.addDock(self.d1)
        self.area.addDock(self.d2, 'right', self.d1)
        self.area.addDock(self.d3, 'right', self.d2)

        # Create layout widgets
        self.w1 = pg.ImageView()
        self.w2 = pg.PlotWidget()
        self.w2.plot()
        self.w2.setLabel('left', 'Intensity', units ='a.u.')
        self.w2.setLabel('bottom', 'time', units ='Frames')

        self.timeStamp_zoom = pg.TextItem(text='')
        self.w1.addItem(self.timeStamp_zoom)
        self.timeStamp_zoom.setPos(0,0)

        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        self.scatter2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))

        self.w1.addItem(self.scatter)
        self.w1.addItem(self.scatter2)

        #options panel
        self.w3 = pg.LayoutWidget()

        self.trackSelector = CheckableComboBox()
        self.tracks= {'None':'None'}
        self.trackSelector.setItems(self.tracks)
        self.trackSelector_label = QLabel("Filter by ID")
        self.selectTrack_checkbox = CheckBox()
        self.selectTrack_checkbox.setChecked(False)
        self.selectTrack_checkbox.stateChanged.connect(self.update)

        self.listAllIDs_checkbox = CheckBox()
        self.listAllIDs_checkbox.setChecked(True)
        self.listAllIDs_checkbox.stateChanged.connect(self.update)
        self.listAllIDs_label = QLabel("List all IDs")

        self.showXaxisLabel_checkbox = CheckBox()
        self.showXaxisLabel_checkbox.setChecked(True)
        self.showXaxisLabel_checkbox.stateChanged.connect(self.update)
        self.showXaxisLabel_label = QLabel("Show X axis label")

        self.showYaxisLabel_checkbox = CheckBox()
        self.showYaxisLabel_checkbox.setChecked(True)
        self.showYaxisLabel_checkbox.stateChanged.connect(self.update)
        self.showYaxisLabel_label = QLabel("Show Y axis label")


        self.displayHist_checkbox = CheckBox()
        self.displayHist_checkbox.setChecked(True)
        self.displayHist_checkbox.stateChanged.connect(self.displayHist)
        self.displayHist_label = QLabel("Show Histogram")

        self.displayTrackPath_checkbox = CheckBox()
        self.displayTrackPath_checkbox.setChecked(True)
        self.displayTrackPath_checkbox.stateChanged.connect(self.update)
        self.displayTrackPath_label = QLabel("Show Track")

        self.trackPathSelector = pg.ComboBox()
        self.pathOptions= {'By Frame':'frame', 'All':'all'}
        self.trackPathSelector.setItems(self.pathOptions)
        self.trackPathSelector.currentIndexChanged.connect(self.update)


        self.displayID_checkbox = CheckBox()
        self.displayID_checkbox.setChecked(False)
        self.displayID_checkbox.stateChanged.connect(self.update)
        self.displayID_label = QLabel("Display Track ID")

        self.displayLegend_checkbox = CheckBox()
        self.displayLegend_checkbox.setChecked(False)
        self.displayLegend_checkbox.stateChanged.connect(self.toggleLegend)
        self.displayLegend_label = QLabel("Display Legend")

        self.displayAxes_checkbox = CheckBox()
        self.displayAxes_checkbox.setChecked(True)
        self.displayAxes_checkbox.stateChanged.connect(self.displayAxes)
        self.displayAxes_label = QLabel("Show Axes")

        self.timeInSec_checkbox = CheckBox()
        self.timeInSec_checkbox.setChecked(False)
        self.timeInSec_checkbox.stateChanged.connect(self.update)
        self.timeInSec_label = QLabel("Time in Seconds")


        # self.interpolate_checkbox = CheckBox()
        # self.interpolate_checkbox.setChecked(False)
        # self.interpolate_checkbox.stateChanged.connect(self.update)
        # self.interpolate_label = QLabel("Interpolate Missing Intensity")

        self.timeStampSize_box = pg.SpinBox(value=50, int=True)
        self.timeStampSize_box.setSingleStep(1)
        self.timeStampSize_box.setMinimum(0)
        self.timeStampSize_box.setMaximum(500)
        self.timeStampSize_label = QLabel("Time Stamp Size")

        self.showTimeStamp_checkbox = CheckBox()
        self.showTimeStamp_checkbox.setChecked(False)
        self.showTimeStamp_checkbox.stateChanged.connect(self.update)
        self.showTimeStamp_label = QLabel("Show Time Stamp")

        self.timeCorrection_checkbox = CheckBox()
        self.timeCorrection_checkbox.setChecked(False)
        self.timeCorrection_checkbox.stateChanged.connect(self.update)
        self.timeCorrection_label = QLabel("Time correction")

        self.timeCorrection_box = pg.SpinBox(value=0, int=True)
        self.timeCorrection_box.setSingleStep(1)
        self.timeCorrection_box.setMinimum(0)
        self.timeCorrection_box.setMaximum(10000)

        self.showScaleBar_button = QPushButton('Scale Bar')
        self.showScaleBar_button.pressed.connect(self.addScaleBar)

        self.showData_button = QPushButton('Load ROI Data')
        self.showData_button.pressed.connect(self.startPlot)

        self.start_box = pg.SpinBox(value=0, int=True)
        self.start_box.setSingleStep(1)
        self.start_box.setMinimum(0)
        self.start_box.setMaximum(10000)
        self.start_label = QLabel("Start Frame")

        self.end_box = pg.SpinBox(value=0, int=True)
        self.end_box.setSingleStep(1)
        self.end_box.setMinimum(0)
        self.end_box.setMaximum(10000)
        self.end_label = QLabel("End Frame")

        self.loop_checkbox = CheckBox()
        self.loop_checkbox.setChecked(False)
        self.loop_label = QLabel("Loop")

        self.play_button = QPushButton('Play/Pause')
        self.play_button.pressed.connect(self.play)

        self.record_button = QPushButton('Export')
        self.record_button.pressed.connect(self.startRecording)

        self.lineCol_Box = pg.ComboBox()
        self.colours = {'trackID':'random', 'green': QColor(Qt.green), 'red': QColor(Qt.red), 'blue': QColor(Qt.blue), 'yellow': QColor(Qt.yellow), 'white': QColor(Qt.white)}
        self.lineCol_Box.setItems(self.colours)

        self.pointCol_Box = pg.ComboBox()
        self.pointCol_Box.setItems(self.colours)
        self.lineCol_label = QLabel("Line color")
        self.pointCol_label = QLabel("Point color")

        self.pointSize_box = pg.SpinBox(value=10, int=True)
        self.pointSize_box.setSingleStep(1)
        self.pointSize_box.setMinimum(0)
        self.pointSize_box.setMaximum(100)

        self.lineWidth_box = pg.SpinBox(value=2, int=True)
        self.lineWidth_box.setSingleStep(1)
        self.lineWidth_box.setMinimum(0)
        self.lineWidth_box.setMaximum(100)

        self.axisLabelSize_box = pg.SpinBox(value=20, int=True)
        self.axisLabelSize_box.setSingleStep(1)
        self.axisLabelSize_box.setMinimum(0)
        self.axisLabelSize_box.setMaximum(1000)

        self.axisTickSize_box = pg.SpinBox(value=20, int=True)
        self.axisTickSize_box.setSingleStep(1)
        self.axisTickSize_box.setMinimum(0)
        self.axisTickSize_box.setMaximum(1000)


        self.labelSpace_label = QLabel("Space offset")
        self.labelSpace_box = pg.SpinBox(value=0, int=True)
        self.labelSpace_box.setSingleStep(1)
        self.labelSpace_box.setMinimum(-1000)
        self.labelSpace_box.setMaximum(1000)

        self.pointSize_label = QLabel("Point Size")
        self.lineWidth_label = QLabel("Line Width")
        self.axisLabelSize_label = QLabel("Axis Label Size")
        self.tickLabelSize_label = QLabel("Axis Tick Label Size")

        self.frameRate_box = pg.SpinBox(value=1, int=True)
        self.frameRate_box.setSingleStep(1)
        self.frameRate_box.setMinimum(0)
        self.frameRate_box.setMaximum(1000)
        self.frameRate_label = QLabel("Frames per sec")

        self.ROISize_checkbox = CheckBox()
        self.ROISize_checkbox.setChecked(False)
        self.ROISize_label = QLabel("Show ROI")
        self.ROISize_box_label = QLabel("ROI size")
        self.ROISize_box = pg.SpinBox(value=3, int=True)
        self.ROISize_box.setSingleStep(1)
        self.ROISize_box.setMinimum(1)
        self.ROISize_box.setMaximum(1000)

        #connect widgets to update
        self.axisTickSize_box.valueChanged.connect(self.update)
        self.axisLabelSize_box.valueChanged.connect(self.update)
        self.pointSize_box.valueChanged.connect(self.update)
        self.lineWidth_box.valueChanged.connect(self.update)
        self.lineCol_Box.currentIndexChanged.connect(self.update)
        self.pointCol_Box.currentIndexChanged.connect(self.update)
        self.frameRate_box.valueChanged.connect(self.update)
        self.timeStampSize_box.valueChanged.connect(self.update)
        self.timeCorrection_box.valueChanged.connect(self.update)
        self.labelSpace_box.valueChanged.connect(self.update)
        self.ROISize_box.valueChanged.connect(self.update)
        self.ROISize_checkbox.stateChanged.connect(self.update)

        # add widgets to layout
        #line + point colour
        self.w3.addWidget(self.lineCol_label, row=0,col=0)
        self.w3.addWidget(self.lineCol_Box, row=0,col=1)
        self.w3.addWidget(self.pointCol_label, row=1,col=0)
        self.w3.addWidget(self.pointCol_Box, row=1,col=1)
        #display/hide hist and labels
        self.w3.addWidget(self.displayHist_label, row=2,col=0)
        self.w3.addWidget(self.displayHist_checkbox, row=2,col=1)
        self.w3.addWidget(self.displayAxes_label, row=2,col=2)
        self.w3.addWidget(self.displayAxes_checkbox, row=2,col=3)

        self.w3.addWidget(self.showXaxisLabel_label, row=3,col=0)
        self.w3.addWidget(self.showXaxisLabel_checkbox, row=3,col=1)
        self.w3.addWidget(self.showYaxisLabel_label, row=3,col=2)
        self.w3.addWidget(self.showYaxisLabel_checkbox, row=3,col=3)

        #line + point size
        self.w3.addWidget(self.lineWidth_label, row=4,col=0)
        self.w3.addWidget(self.lineWidth_box, row=4,col=1)
        self.w3.addWidget(self.pointSize_label, row=4,col=2)
        self.w3.addWidget(self.pointSize_box, row=4,col=3)

        #label size
        self.w3.addWidget(self.axisLabelSize_label, row=5,col=0)
        self.w3.addWidget(self.axisLabelSize_box, row=5,col=1)

        self.w3.addWidget(self.tickLabelSize_label, row=6,col=0)
        self.w3.addWidget(self.axisTickSize_box, row=6,col=1)

        #label space offset
        self.w3.addWidget(self.labelSpace_label, row=7,col=0)
        self.w3.addWidget(self.labelSpace_box, row=7,col=1)

        #select track
        #display track for all frames
        self.w3.addWidget(self.listAllIDs_label, row=8,col=0, colspan=2)
        self.w3.addWidget(self.listAllIDs_checkbox, row=8,col=2)
        self.w3.addWidget(self.trackSelector_label, row=9,col=0)
        self.w3.addWidget(self.selectTrack_checkbox, row=9,col=1)
        self.w3.addWidget(self.trackSelector, row=9,col=2)

        #set time unit
        self.w3.addWidget(self.timeInSec_label, row=10,col=0)
        self.w3.addWidget(self.timeInSec_checkbox , row=10,col=1)


        #Display ROI
        self.w3.addWidget(self.ROISize_label, row=11,col=0)
        self.w3.addWidget(self.ROISize_checkbox, row=11,col=1)
        self.w3.addWidget(self.ROISize_box_label, row=11,col=2)
        self.w3.addWidget(self.ROISize_box, row=11,col=3)

        #show track path
        self.w3.addWidget(self.displayTrackPath_label, row=12,col=0)
        self.w3.addWidget(self.displayTrackPath_checkbox, row=12,col=1)
        self.w3.addWidget(self.trackPathSelector, row=12,col=2)

        #display track ID
        self.w3.addWidget(self.displayID_label, row=13,col=0)
        self.w3.addWidget(self.displayID_checkbox, row=13,col=1)
        #display Legend
        self.w3.addWidget(self.displayLegend_label, row=14,col=0)
        self.w3.addWidget(self.displayLegend_checkbox, row=14,col=1)

        #show time stamp
        self.w3.addWidget(self.timeStampSize_label , row=15,col=0)
        self.w3.addWidget(self.timeStampSize_box, row=15,col=1)
        self.w3.addWidget(self.showTimeStamp_label , row=16,col=0)
        self.w3.addWidget(self.showTimeStamp_checkbox, row=16,col=1)

        #time correction
        self.w3.addWidget(self.timeCorrection_label , row=17,col=0)
        self.w3.addWidget(self.timeCorrection_checkbox, row=17,col=1)
        self.w3.addWidget(self.timeCorrection_box, row=17,col=2)

        #show scale bar
        self.w3.addWidget(self.showScaleBar_button, row=18,col=0)
        #play
        self.w3.addWidget(self.start_label, row=19,col=0)
        self.w3.addWidget(self.start_box, row=19,col=1)
        self.w3.addWidget(self.end_label, row=19,col=2)
        self.w3.addWidget(self.end_box, row=19,col=3)

        self.w3.addWidget(self.frameRate_label, row=20,col=0)
        self.w3.addWidget(self.frameRate_box, row=20,col=1)
        self.w3.addWidget(self.play_button, row=20,col=2)
        #showData
        self.w3.addWidget(self.showData_button, row=21,col=0)
        #record
        self.w3.addWidget(self.record_button, row=21,col=1)

        #add layouts to dock
        self.d1.addWidget(self.w1)
        self.d2.addWidget(self.w2)
        self.d3.addWidget(self.w3)

        #hide imageview buttons
        self.w1.ui.roiBtn.hide()
        self.w1.ui.menuBtn.hide()

    def update(self):

        #update axis labels
        labelStyle = {'color': '#FFF', 'font-size': '{}pt'.format(self.axisLabelSize_box.value())}
        # Set the font size for the x-axis label
        self.w2.getAxis("left").setLabel('Intensity', units='a.u.', **labelStyle)
        # Set the font size for the y-axis label
        if self.timeInSec_checkbox.isChecked():
            self.w2.getAxis("bottom").setLabel('Time', units='s', **labelStyle)
        else:
            self.w2.getAxis("bottom").setLabel('Time', units='Frames', **labelStyle)

        if self.showXaxisLabel_checkbox.isChecked():
            self.w2.getAxis("left").showLabel(show=True)
        else:
            self.w2.getAxis("left").showLabel(show=False)

        if self.showYaxisLabel_checkbox.isChecked():
            self.w2.getAxis("bottom").showLabel(show=True)
        else:
            self.w2.getAxis("bottom").showLabel(show=False)

        font = QFont()
        font.setPixelSize(self.axisTickSize_box.value())
        self.w2.getAxis("left").setStyle(tickFont = font)
        self.w2.getAxis("bottom").setStyle(tickFont =font)

        self.w2.getAxis("left").setStyle(tickTextOffset = self.axisTickSize_box.value())
        self.w2.getAxis("bottom").setStyle(tickTextOffset = self.axisTickSize_box.value())

        self.w2.getAxis("left").setWidth(self.labelSpace_box.value() + 60 + self.axisLabelSize_box.value() + (1.8*self.axisTickSize_box.value()))
        self.w2.getAxis("bottom").setHeight(self.labelSpace_box.value() + 40 + self.axisLabelSize_box.value() + (1.8*self.axisTickSize_box.value()))

        #test if roiZoom plot intiated
        if self.dataWindow == None:
            return
        #clear scatter plot data
        self.scatter.setData([],[])
        self.scatter2.setData([],[])
        #clear intensity plot
        self.w2.clear()

        #get current frame number
        frame = self.dataWindow.currentIndex

        #get old histogram levels
        if self.ROIplotInitiated == False:
            #get mainwindow level for 1st plot
            hist_levels = self.dataWindow.imageview.getHistogramWidget().getLevels()
        else:
            hist_levels = self.w1.getLevels()

        #set ROI zoom image
        self.w1.setImage(self.dataWindow.currentROI.getArrayRegion(self.array[frame], self.dataWindow.imageview.imageItem, autoLevels=False))

        #set mx and my
        self.mx, self.my = self.dataWindow.currentROI.size()

        #update hist
        self.w1.setLevels(min=hist_levels[0],max=hist_levels[1])

        #self.w1.autoRange()
        roiShape = self.currentROI.mapToItem(self.dataWindow.scatterPlot, self.currentROI.shape())
        # Get list of all points inside shape
        self.selectedPoints = [[frame, pt.x(), pt.y()] for pt in self.mainGUI.getScatterPointsAsQPoints() if roiShape.contains(pt)]
        self.getDataFromScatterPoints()

        if self.ROIplotInitiated == False:
            self.ROIplotInitiated = True

        #determine which tracks to display
        if self.selectTrack_checkbox.isChecked():
            tracksToDisplay = np.array(self.trackSelector.value(), dtype=int)
            self.pointsToDisplay = [x for x in self.dataInView if x[0] in tracksToDisplay]
        else:
            tracksToDisplay = self.tracksInView
            self.pointsToDisplay = self.dataInView

        #print(self.dataInView)

        #get ROI pos to offset scatter plot on zoomed image
        pos = self.currentROI.pos()

        #plot points in ROI on zoomed ROI image
        if len(self.pointsToDisplay) != 0:
            trackID_list = np.array(extractListElement(self.pointsToDisplay, 0))
            x_list = np.array(extractListElement(self.pointsToDisplay, 1)) - pos[0]
            y_list = np.array(extractListElement(self.pointsToDisplay, 2)) - pos[1]

            if self.pointCol_Box.value() == 'random':
                trackColour_list = [pg.intColor(i) for i in trackID_list]
            else:
                trackColour_list = [self.pointCol_Box.value() for i in trackID_list]


            brush_list = [pg.mkBrush(i) for i in trackColour_list]
            pen_list = [pg.mkPen(i, width=5) for i in trackColour_list]

            self.scatter.addPoints(x=x_list, y=y_list, size=self.pointSize_box.value(), brush=brush_list, name=trackID_list, hoverable=False)

            #plot boxs representing ROIs used to get mean intensity
            if self.ROISize_checkbox.isChecked():

                if self.ROISize_box.value() % 2 == 0:
                    x_rounded_list = np.around(x_list).astype(int)
                    y_rounded_list = np.around(y_list).astype(int)
                else:
                    x_rounded_list = x_list.astype(int) + 0.5
                    y_rounded_list = y_list.astype(int) + 0.5

                self.scatter2.addPoints(x=x_rounded_list, y=y_rounded_list, size=self.ROISize_box.value(), pen=pen_list, brush=None, symbol='s',pxMode=False, hoverable=False)


            if self.displayID_checkbox.isChecked():
                label_list = [custom_symbol(str(i)) for i in trackID_list]
                offset = self.pointSize_box.value()/10
                self.scatter.addPoints(x=x_list+offset, y=y_list+offset, size=self.pointSize_box.value(), symbol = label_list, brush=brush_list, name=trackID_list, hoverable=False)


        #add timestamp
        if self.timeCorrection_checkbox.isChecked():
            correctedFrame = frame - self.timeCorrection_box.value()
        else:
            correctedFrame = frame

        if self.timeInSec_checkbox.isChecked():
            timestamp = round((correctedFrame * self.mainGUI.trackPlotOptions.frameLength_selector.value()) /1000, 2)
            time_text = '{0:.2f} s'.format(timestamp)
        else:
            timestamp = correctedFrame
            time_text = str(timestamp)

        if self.showTimeStamp_checkbox.isChecked():

            font_size = str(self.timeStampSize_box.value())
            font_style = 'bold'
            html="<span style='font-size: {}pt; font-style: {};'>{}</span>".format(font_size, font_style, time_text)
            self.timeStamp_zoom.setHtml(html)
        else:
            self.timeStamp_zoom.setText('')


        #print(self.trackSelector.value())

        #plot intensity trace(s)
        for trackID in tracksToDisplay:
            # Get data for the selected track
            trackDF = self.mainGUI.data[self.mainGUI.data['track_number'] == trackID]

            #filter to current frame
            trackDF = trackDF[trackDF['frame'] <= frame]

            #intensity trace choice from trackPlot options
            intensity = trackDF[self.mainGUI.trackPlotOptions.intensityChoice_Box.value()].to_numpy()
            #use background subtracted intensity if option selected
            if self.mainGUI.trackPlotOptions.backgroundSubtract_checkbox.isChecked():
                intensity = intensity - self.mainGUI.trackPlotOptions.background_selector.value()


            if self.timeInSec_checkbox.isChecked():
                xData = (trackDF['frame'].to_numpy() * self.mainGUI.trackPlotOptions.frameLength_selector.value()) /1000
                if self.timeCorrection_checkbox.isChecked():
                    xData = xData - (self.timeCorrection_box.value() * self.mainGUI.trackPlotOptions.frameLength_selector.value() /1000)
            else:
                xData = trackDF['frame'].to_numpy()
                if self.timeCorrection_checkbox.isChecked():
                    xData = xData - self.timeCorrection_box.value()



            item = pg.PlotDataItem(x=xData,y=intensity, name=str(trackID))
            # Map the trackID to a colour
            if self.lineCol_Box.value() == 'random':
                trackColour = pg.intColor(trackID)
            else:
                trackColour = self.lineCol_Box.value()
            # setup pen
            pen = pg.mkPen(trackColour, width=self.lineWidth_box.value())
            item.setPen(pen)
            self.w2.addItem(item)

        #plot track on zoom display
        if self.displayTrackPath_checkbox.isChecked():
            self.plotTracks(tracksToDisplay)
        else:
            self.clearTracks()


    def startPlot(self):
        self.dataWindow = self.mainGUI.plotWindow

        if self.dataWindow.currentROI == None:
            g.messageBox('Warning','First draw ROI on Main Display')
            return

        self.array = self.dataWindow.imageArray()
        self.nFrames = self.dataWindow.mt

        #self.zoomIMG = pg.ImageItem()
        #self.w1.addItem(self.zoomIMG)

        self.currentROI = self.dataWindow.currentROI
        self.currentROI.sigRegionChanged.connect(self.update)

        self.dataWindow.sigTimeChanged.connect(self.update)

        #set axis limits to span all frames
        self.w2.setXRange(0, self.dataWindow.mt)
        self.w2.setLimits(xMin=0, xMax=self.dataWindow.mt)

        self.start_box.setMinimum(0)
        self.start_box.setMaximum(self.dataWindow.mt)

        self.end_box.setMinimum(0)
        self.end_box.setMaximum(self.dataWindow.mt)
        self.end_box.setValue(self.dataWindow.mt)

        self.timeCorrection_box.setMaximum(self.dataWindow.mt)

        self.update()

        # Check if user wants to plot a specific track or use the display track
        if self.selectTrack_checkbox.isChecked():
            self.trackList  = [int(self.trackSelector.value())]
        else:
            self.trackList = [self.trackSelector.itemText(i) for i in range(self.trackSelector.count())]


    def toggleLegend(self):
        if self.displayLegend_checkbox.isChecked():
            if self.traceLegend != None:
                self.traceLegend = self.w2.addLegend()
        else:
            self.w2.removeItem(self.traceLegend)

        self.update()


    def updateTrackList(self):
        """
        Update the track list displayed in the GUI based on the data loaded into the application.
        """
        if self.mainGUI.useFilteredData == False:
            self.tracks = dictFromList(self.mainGUI.data['track_number'].to_list())  # Convert a column of track numbers into a dictionary
        else:
            self.tracks = dictFromList(self.mainGUI.filteredData['track_number'].to_list())  # Convert a column of track numbers into a dictionary


        self.trackSelector.setItems(self.tracks)  # Set the track list in the GUI

    def getDataFromScatterPoints(self):
        # Get track IDs for all points in scatter plot
        self.tracksInView  = []
        self.dataInView = []

        # Flatten scatter plot data into a single list of points
        #flat_ptList = [pt for sublist in self.selectedPoints for pt in sublist]
        flat_ptList = self.selectedPoints

        # Loop through each point and get track IDs for corresponding data points in DataFrame
        for pt in flat_ptList:
            #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))

            ptFilterDF = self.mainGUI.data[(self.mainGUI.data['x']==pt[1]) & (self.mainGUI.data['y']==pt[2])]

            self.tracksInView.extend([ptFilterDF['track_number'].values[0]])

            #intensity trace choice from trackPlot options
            intensity = ptFilterDF[self.mainGUI.trackPlotOptions.intensityChoice_Box.value()].to_numpy()
            #use background subtracted intensity if option selected
            if self.mainGUI.trackPlotOptions.backgroundSubtract_checkbox.isChecked():
                intensity = intensity - self.mainGUI.trackPlotOptions.background_selector.value()

            self.dataInView.append([ptFilterDF['track_number'].values[0],ptFilterDF['x'].values[0],ptFilterDF['y'].values[0], intensity[0]])

        if self.listAllIDs_checkbox.isChecked():
            self.updateTrackList()
        else:
            self.trackSelector.setItems(dictFromList(self.tracksInView))


    # def getInterpolatedPoints(self):
    #     # Extract x,y,frame data for each point
    #     points = np.column_stack((trackDF['frame'].to_list(), trackDF['x'].to_list(), trackDF['y'].to_list()))

    #     if self.interpolate_checkbox.isChecked():
    #         #interpolate points for missing frames
    #         allFrames = range(int(min(points[:,0])), int(max(points[:,0]))+1)
    #         xinterp = np.interp(allFrames, points[:,0], points[:,1])
    #         yinterp = np.interp(allFrames, points[:,0], points[:,2])

    #         points = np.column_stack((allFrames, xinterp, yinterp))


    #     # Loop through each point and extract a cropped image
    #     for point in points:
    #         minX = round(point[1]) - x_limit + self.d # Determine the limits of the crop including padding
    #         maxX = round(point[1]) + x_limit + self.d
    #         minY = round(point[2]) - y_limit + self.d
    #         maxY = round(point[2]) + y_limit + self.d

    #         if (self.d % 2) == 0:
    #             crop = self.A_pad[int(point[0]),minX:maxX,minY:maxY] - np.min(self.A[int(point[0])])# Extract the crop
    #         else:
    #             crop = self.A_pad[int(point[0]),minX-1:maxX,minY-1:maxY] - np.min(self.A[int(point[0])])# Extract the crop

    #         A_crop[int(point[0])] = crop

    #     self.A_crop_stack[i] = A_crop # Store the crop in the array of cropped images

    #     A_crop[A_crop==0] = np.nan
    #     trace = np.mean(A_crop, axis=(1,2))



    def plotTracks(self, trackIDs):
        '''Updates track paths '''
        pos = self.currentROI.pos()
        frame = self.dataWindow.currentIndex

        # clear self.pathitems
        self.clearTracks()

        for trackID in trackIDs:
            trackDF = self.mainGUI.data[self.mainGUI.data['track_number'] == trackID]

            if self.trackPathSelector.value() == 'frame':
                #filter to current frame
                trackDF = trackDF[trackDF['frame'] <= frame]

            pathitem = QGraphicsPathItem(self.w1.view)

            # Map the trackID to a colour
            if self.lineCol_Box.value() == 'random':
                trackColour = pg.intColor(trackID)
            else:
                trackColour = self.lineCol_Box.value()

            pen = pg.mkPen(trackColour, width=2)

            # set the pen for the path items
            pathitem.setPen(pen)

            # add the path items to the view(s)
            self.w1.view.addItem(pathitem)

            # keep track of the path items
            self.pathitems.append(pathitem)

            # extract the x and y coordinates for the track
            x = trackDF['x'].to_numpy() - pos[0]
            y = trackDF['y'].to_numpy() - pos[1]

            # create a QPainterPath for the track and set the path for the path item
            path = QPainterPath(QPointF(x[0],y[0]))

            for i in np.arange(1, len(x)):
                path.lineTo(QPointF(x[i],y[i]))

            pathitem.setPath(path)


    def clearTracks(self):
        # Check that there is an open plot window
        if self.w1 is not None:
            # Remove each path item from the plot window
            for pathitem in self.pathitems:
                self.w1.view.removeItem(pathitem)

        # Reset the path items list to an empty list
        self.pathitems = []



    def play(self):
        if self.dataWindow.currentIndex >= self.end_box.value()-1:
            self.dataWindow.setIndex(0)

        if self.end_box.value() < self.dataWindow.mt:
            self.dataWindow.sigTimeChanged.connect(self.timeLineChange)

        if self.dataWindow.imageview.playTimer.isActive():
            self.dataWindow.imageview.play(0)
        else:
            if self.dataWindow.currentIndex < self.start_box.value():
                self.dataWindow.setIndex(self.start_box.value())
            self.dataWindow.imageview.play(int(self.frameRate_box.value()))

    def timeLineChange(self):
        #print(self.dataWindow.currentIndex)
        if self.dataWindow.currentIndex+1 > self.end_box.value():
            if self.dataWindow.imageview.playTimer.isActive():
                self.dataWindow.imageview.play(0)
                self.dataWindow.sigTimeChanged.disconnect(self.timeLineChange)


    def startRecording(self):
        ## Check if ffmpeg is installed
        if os.name == 'nt':  # If we are running windows
            try:
                subprocess.call(["ffmpeg"])
            except FileNotFoundError as e:
                if e.errno == errno.ENOENT:
                    # handle file not found error.
                    # I used http://ffmpeg.org/releases/ffmpeg-2.8.4.tar.bz2 originally
                    g.alert("The program FFmpeg is required to export movies. \
                    \n\nFor instructions on how to install, go here: http://www.wikihow.com/Install-FFmpeg-on-Windows")
                    return None
                else:
                    # Something else went wrong while trying to run `wget`
                    raise

        filetypes = "Movies (*.mp4)"
        prompt = "Save movie to .mp4 file"
        filename = save_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None

        exporter0 = pg.exporters.ImageExporter(self.dataWindow.imageview.view)
        exporter1 = pg.exporters.ImageExporter(self.w1.view)
        exporter2 = pg.exporters.ImageExporter(self.w2.plotItem)

        tmpdir = os.path.join(os.path.dirname(g.settings.settings_file), 'tmp')
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
        os.mkdir(tmpdir)

        subDir_list = [['main',os.path.join(tmpdir, 'main')],
                       ['zoom',os.path.join(tmpdir, 'zoom')],
                       ['trace',os.path.join(tmpdir, 'trace')]]

        for d in subDir_list:
            os.makedirs(d[1])


        for n,i in enumerate(range(self.start_box.value(),self.end_box.value())):
            self.dataWindow.setIndex(i)
            exporter0.export(os.path.join(os.path.join(tmpdir, 'main'), '{:03}.jpg'.format(n)))
            exporter1.export(os.path.join(os.path.join(tmpdir, 'zoom'), '{:03}.jpg'.format(n)))
            exporter2.export(os.path.join(os.path.join(tmpdir, 'trace'), '{:03}.jpg'.format(n)))
            qApp.processEvents()

        print('temp movie files saved to {}'.format(tmpdir))

        rate = int(self.frameRate_box.value())

        olddir = os.getcwd()
        print('movie directory: {}'.format(olddir))

        for d in subDir_list:

            os.chdir(d[1])
            subprocess.call(
                ['ffmpeg', '-r', '%d' % rate, '-i', '%03d.jpg', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', 'output.mp4'])
            split = os.path.splitext(filename)
            movieName = split[0] + '_' + d[0] + split[1]
            os.rename('output.mp4', movieName)
            os.chdir(olddir)
            print('Successfully saved movie as {}.'.format(os.path.basename(movieName)))
            g.m.statusBar().showMessage('Successfully saved movie as {}.'.format(os.path.basename(movieName)))


    def addScaleBar(self):
        self.scale_bar.gui()

    def displayHist(self):
        if self.displayHist_checkbox.isChecked():
            self.w1.ui.histogram.show()
            #self.w1.ui.roiBtn.show()
            #self.w1.ui.menuBtn.show()
        else:
            self.w1.ui.histogram.hide()
            #self.w1.ui.roiBtn.hide()
            #self.w1.ui.menuBtn.hide()


    def displayAxes(self):
        if self.displayAxes_checkbox.isChecked():
            self.w2.getPlotItem().showAxis('bottom')
            self.w2.getPlotItem().showAxis('left')
        else:
            self.w2.getPlotItem().hideAxis('bottom')
            self.w2.getPlotItem().hideAxis('left')
        #self.w2.show()

    def show(self):
        """
        Shows the main window.
        """
        self.win.show()

    def close(self):
        """
        Closes the main window.
        """
        self.win.close()

    def hide(self):
        """
        Hides the main window.
        """
        self.win.hide()

