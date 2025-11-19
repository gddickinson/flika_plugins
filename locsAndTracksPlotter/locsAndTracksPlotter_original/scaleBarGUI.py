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

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui


class Scale_Bar_ROIzoom(BaseProcess):
    ''' scale_bar(width_NoUnits, width_pixels, font_size, color, background, location, show=True)

    Parameters:
        width_NoUnits (float): width
        width_pixels (float): width in pixels
        font_size (int): size of the font
        color (string): ['Black', White']
        background (string): ['Black','White', 'None']
        location (string): ['Lower Right','Lower Left','Top Right','Top Left']
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

        font_size.setValue(12)
        width_pixels.setValue(1.00)
        width_NoUnits.setValue(108)

        show.setChecked(True)
        self.items.append({'name':'width_NoUnits','string':'Width of bar','object':width_NoUnits})
        self.items.append({'name':'width_unit','string':'Width of bar units','object':units})
        self.items.append({'name':'width_pixels','string':'Width of bar in pixels','object':width_pixels})
        self.items.append({'name':'font_size','string':'Font size','object':font_size})
        self.items.append({'name':'color','string':'Color','object':color})
        self.items.append({'name':'background','string':'Background','object':background})
        self.items.append({'name':'location','string':'Location','object':location})
        self.items.append({'name':'show','string':'Show','object':show})

        super().gui()
        self.preview()

    def __call__(self,width_NoUnits, width_pixels, font_size, color, background,location,show=True,keepSourceWindow=None):

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
                barPoint=QPoint(0, textRect.height())
            elif location=='Top Right':
                barPoint=QPoint(-width_pixels, textRect.height())
            elif location=='Lower Right':
                barPoint=QPoint(-width_pixels, -textRect.height())
            elif location=='Lower Left':
                barPoint=QPoint(0, -textRect.height())

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
        view = self.w.view
        textRect=self.roiGUI.scaleBarLabel.boundingRect()
        textWidth=textRect.width()*view.viewPixelSize()[0]
        textHeight=textRect.height()*view.viewPixelSize()[1]

        if location=='Top Left':
            barPoint=QPoint(0, 1.3*textHeight)
            self.roiGUI.scaleBarLabel.setPos(QPointF(width_pixels/2-textWidth/2,0))
        elif location=='Top Right':
            barPoint=QPoint(self.roiGUI.mx-width_pixels, 1.3*textHeight)
            self.roiGUI.scaleBarLabel.setPos(QPointF(self.roiGUI.mx-width_pixels/2-textWidth/2,0))
        elif location=='Lower Right':
            barPoint=QPoint(self.roiGUI.mx-width_pixels, self.roiGUI.my-1.3*textHeight)
            self.roiGUI.scaleBarLabel.setPos(QPointF(self.roiGUI.mx-width_pixels/2-textWidth/2,self.roiGUI.my-textHeight))
        elif location=='Lower Left':
            barPoint=QPoint(0, self.roiGUI.my-1.3*textHeight)
            self.roiGUI.scaleBarLabel.setPos(QPointF(QPointF(width_pixels/2-textWidth/2,self.roiGUI.my-textHeight)))
        self.roiGUI.scaleBarLabel.bar.setRect(QRectF(barPoint, QSizeF(width_pixels,textHeight/4)))

    def preview(self):
        width_NoUnits=self.getValue('width_NoUnits')
        width_pixels=self.getValue('width_pixels')
        font_size=self.getValue('font_size')
        color=self.getValue('color')
        background=self.getValue('background')
        location=self.getValue('location')
        show=self.getValue('show')
        self.__call__(width_NoUnits, width_pixels, font_size, color, background, location, show)
