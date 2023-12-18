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
        width_NoUnits.setValue(self.roiGUI.pixelSize_box.value())

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


class ROIPLOT():
    """
    A class for displaying ROI image with scrolling update of locs positions and intensity trace.
    """
    def __init__(self, mainGUI):
        super().__init__()

        # Initialize the main GUI instance
        self.mainGUI = mainGUI
        self.dataWindow = None
        self.ROIplotInitiated = False
        self.mx = None
        self.my = None
        self.nFrames = None
        self.scaleBarLabel = None

        self.scale_bar = Scale_Bar_ROIzoom(self)

        # Create a dock window and add a dock
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1400,700)
        self.win.setWindowTitle('ROI Video Exporter')
        self.d1 = Dock("ROI Zoom Window", size=(400, 700))
        self.d3 = Dock("Options Panel", size=(200, 700))
        self.area.addDock(self.d1)
        self.area.addDock(self.d3, 'right', self.d1)

        # Create layout widgets
        self.w1 = pg.ImageView()
        self.w2 = pg.PlotWidget()

        self.timeStamp_zoom = pg.TextItem(text='')
        self.w1.addItem(self.timeStamp_zoom)
        self.timeStamp_zoom.setPos(0,0)


        #options panel
        self.w3 = pg.LayoutWidget()

        self.pixelSize_box = pg.SpinBox(value=self.mainGUI.pixelSize.value(), int=True)
        self.pixelSize_box.setSingleStep(1)
        self.pixelSize_box.setMinimum(0)
        self.pixelSize_box.setMaximum(1000)
        self.pixelSize_label = QLabel("Pixel Size (nm)")

        self.framelength_box = pg.SpinBox(value=self.mainGUI.framelength.value(), int=True)
        self.framelength_box.setSingleStep(1)
        self.framelength_box.setMinimum(0)
        self.framelength_box.setMaximum(10000)
        self.framelength_label = QLabel("Frame Length (ms)")

        self.timeInSec_checkbox = CheckBox()
        self.timeInSec_checkbox.setChecked(False)
        self.timeInSec_checkbox.stateChanged.connect(self.update)
        self.timeInSec_label = QLabel("Time in Seconds")

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

        self.frameRate_box = pg.SpinBox(value=1, int=True)
        self.frameRate_box.setSingleStep(1)
        self.frameRate_box.setMinimum(0)
        self.frameRate_box.setMaximum(1000)
        self.frameRate_label = QLabel("Frames per sec")

        #connect widgets to update
        self.frameRate_box.valueChanged.connect(self.update)
        self.timeStampSize_box.valueChanged.connect(self.update)
        self.timeCorrection_box.valueChanged.connect(self.update)

        # add widgets to layout

        #pixel size
        self.w3.addWidget(self.pixelSize_label , row=0,col=0)
        self.w3.addWidget(self.pixelSize_box, row=0,col=1)

        #framelength
        self.w3.addWidget(self.framelength_label , row=1,col=0)
        self.w3.addWidget(self.framelength_box, row=1,col=1)

        #set time unit
        self.w3.addWidget(self.timeInSec_label, row=10,col=0)
        self.w3.addWidget(self.timeInSec_checkbox , row=10,col=1)

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
        self.d3.addWidget(self.w3)

        #hide imageview buttons
        self.w1.ui.roiBtn.hide()
        self.w1.ui.menuBtn.hide()

        #load ROI data
        self.startPlot(startup=True)

    def update(self):

        #test if roiZoom plot intiated
        if self.dataWindow == None:
            return


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

        #add timestamp
        if self.timeCorrection_checkbox.isChecked():
            correctedFrame = frame - self.timeCorrection_box.value()
        else:
            correctedFrame = frame

        if self.timeInSec_checkbox.isChecked():
            timestamp = round((correctedFrame * self.framelength_box.value()) /1000, 2)
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

        #update plotInitiated flag
        if self.ROIplotInitiated == False:
            self.ROIplotInitiated = True


    def startPlot(self, startup=False):
        self.dataWindow = self.mainGUI.plotWindow

        if self.dataWindow.currentROI == None:
            if startup:
                return
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

        tmpdir = os.path.join(os.path.dirname(g.settings.settings_file), 'tmp')
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
        os.mkdir(tmpdir)

        subDir_list = [['main',os.path.join(tmpdir, 'main')],
                       ['zoom',os.path.join(tmpdir, 'zoom')]]

        for d in subDir_list:
            os.makedirs(d[1])


        for n,i in enumerate(range(self.start_box.value(),self.end_box.value())):
            self.dataWindow.setIndex(i)
            exporter0.export(os.path.join(os.path.join(tmpdir, 'main'), '{:03}.jpg'.format(n)))
            exporter1.export(os.path.join(os.path.join(tmpdir, 'zoom'), '{:03}.jpg'.format(n)))
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

