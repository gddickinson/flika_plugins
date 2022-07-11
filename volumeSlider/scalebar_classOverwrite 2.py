# -*- coding: utf-8 -*-
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore, QtGui
import flika
from distutils.version import StrictVersion
#from .. import global_vars as g

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector


class Scale_Bar_volumeView(BaseProcess):
    ''' scale_bar(unit,width_units, width_pixels,font_size, color, background,location,orientation,show=True,show_label=True)

    Parameters:
        unit (string): ['micro','nano','pixels']
        width_units (float): width displayed in label
        width_pixels (float): width in pixels of scale bar
        font_size (int): size of the font
        color (string): ['Black', White']
        background (string): ['Black','White', 'None']
        location (string): ['Lower Right','Lower Left','Top Right','Top Left']
        show (bool): controls whether the Scale_bar is displayed or not
        show_label (bool): controls whether the Scale_bar label is displayed or not
    '''
    
    def __init__(self, w, height, width):
        super().__init__()
        self.w = w
        self.height = height
        self.width = width
        
    def gui(self):
        self.gui_reset()
        #w=g.win
        w = self.w
        width_units=QtWidgets.QDoubleSpinBox()
        
        width_pixels=QtWidgets.QSpinBox()
        width_units.setRange(.001,1000000)
        width_pixels.setRange(1,self.width)
        
        font_size=QtWidgets.QSpinBox()

        unit=ComboBox()
        unit.addItem('micro')
        unit.addItem('nano')
        unit.addItem('pixels')
        
        orientation=ComboBox()
        orientation.addItem('horizontal')
        orientation.addItem('vertical')
        
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
        show_label=CheckBox()        
        
        if hasattr(w,'scaleBarLabel') and w.scaleBarLabel is not None: #if the scaleBarLabel already exists
            props=w.scaleBarLabel.flika_properties
            width_units.setValue(props['width_units'])
            width_pixels.setValue(props['width_pixels'])
            unit.setCurrentIndex(color.findText(props['unit']))
            orientation.setCurrentIndex(color.findText(props['orientation']))
            font_size.setValue(props['font_size'])
            color.setCurrentIndex(color.findText(props['color']))
            background.setCurrentIndex(background.findText(props['background']))
            location.setCurrentIndex(location.findText(props['location']))
        else:
            font_size.setValue(12)
            width_pixels.setValue(int(w.view.width()/8))
            width_units.setValue(1)
            
        show.setChecked(True) 
        show_label.setChecked(True)
                        
        self.items.append({'name':'unit','string':'Units','object':unit})  
        self.items.append({'name':'width_units','string':'Width of bar in [Units]','object':width_units})
        self.items.append({'name':'width_pixels','string':'Width of bar in pixels','object':width_pixels})      
        self.items.append({'name':'font_size','string':'Font size','object':font_size})
        self.items.append({'name':'color','string':'Color','object':color})
        self.items.append({'name':'background','string':'Background','object':background})
        self.items.append({'name':'location','string':'Location','object':location})
        self.items.append({'name':'orientation','string':'Orientation','object':orientation})        
        self.items.append({'name':'show','string':'Show','object':show})
        self.items.append({'name':'show_label','string':'Show label','object':show_label})        
        
        super().gui()
        self.preview()

    def __call__(self,unit,width_units, width_pixels,font_size, color, background,location,orientation,show=True,show_label=True,keepSourceWindow=None):
        #w=g.win
        w = self.w
        if show:
            if hasattr(w,'scaleBarLabel') and w.scaleBarLabel is not None:
                w.view.removeItem(w.scaleBarLabel.bar)
                w.view.removeItem(w.scaleBarLabel)
                try:
                    w.view.sigResized.disconnect(self.updateBar)
                except:
                    pass
            if location=='Top Left':
                anchor=(0,0)
                pos=[0,0]
            elif location=='Top Right':
                anchor=(0,0)
                pos=[self.width,0]
            elif location=='Lower Right':
                anchor=(0,0)
                pos=[self.width,self.height]
            elif location=='Lower Left':
                anchor=(0,0)
                pos=[0,self.height]
                
            if unit=='micro':
                unitText = 'μm'
            elif unit=='nano':
                unitText = 'nm'
            elif unit=='pixels':
                unitText = 'px'
                                
                
            w.scaleBarLabel= pg.TextItem(anchor=anchor, html="<span style='font-size: {}pt;color:{};background-color:{};'>{} {}</span>".format(font_size, color, background,width_units,unitText))
            w.scaleBarLabel.setPos(pos[0],pos[1])
            w.scaleBarLabel.flika_properties={item['name']:item['value'] for item in self.items}
            w.view.addItem(w.scaleBarLabel)
            if color=='White':
                color255=[255,255,255,255]
            elif color=='Black':
                color255=[0,0,0,255]
            textRect=w.scaleBarLabel.boundingRect()
            
            if location=='Top Left':
                barPoint=QtCore.QPoint(0, textRect.height())
            elif location=='Top Right':
                barPoint=QtCore.QPoint(-width_pixels, textRect.height())
            elif location=='Lower Right':
                barPoint=QtCore.QPoint(-width_pixels, -textRect.height())
            elif location=='Lower Left':
                barPoint=QtCore.QPoint(0, -textRect.height())
            
            if orientation=='horizontal':
                bar = QtWidgets.QGraphicsRectItem(QtCore.QRectF(barPoint, QtCore.QSizeF(width_pixels,int(font_size/3))))
            elif orientation=='vertical':
                bar = QtWidgets.QGraphicsRectItem(QtCore.QRectF(barPoint, QtCore.QSizeF(int(font_size/3),width_pixels)))
            bar.setPen(pg.mkPen(color255)); bar.setBrush(pg.mkBrush(color255))
            w.view.addItem(bar)
            #bar.setParentItem(w.scaleBarLabel)
            w.scaleBarLabel.bar=bar
            w.view.sigResized.connect(self.updateBar)
            self.updateBar()
            
        else:
            if hasattr(w,'scaleBarLabel') and w.scaleBarLabel is not None:
                w.view.removeItem(w.scaleBarLabel.bar)
                w.view.removeItem(w.scaleBarLabel)
                w.scaleBarLabel=None
                w.view.sigResized.disconnect(self.updateBar)
        return None
        
    def updateBar(self):
        #w=g.win
        w = self.w
        width_pixels=self.getValue('width_pixels')
        location=self.getValue('location')
        orientation=self.getValue('orientation')
        view = w.view
        textRect=w.scaleBarLabel.boundingRect()
        textWidth=textRect.width()*view.viewPixelSize()[0]
        textHeight=textRect.height()*view.viewPixelSize()[1]
        show_label = self.getValue('show_label')
        
        if location=='Top Left':
            barPoint=QtCore.QPoint(0, 1.3*textHeight)
            w.scaleBarLabel.setPos(QtCore.QPointF(width_pixels/2-textWidth/2,0))
        elif location=='Top Right':
            barPoint=QtCore.QPoint(self.width-width_pixels, 1.3*textHeight)
            w.scaleBarLabel.setPos(QtCore.QPointF(self.width-width_pixels/2-textWidth/2,0))
        elif location=='Lower Right':
            barPoint=QtCore.QPoint(self.width-width_pixels, self.height-1.3*textHeight)
            w.scaleBarLabel.setPos(QtCore.QPointF(self.width-width_pixels/2-textWidth/2,self.height-textHeight))
        elif location=='Lower Left':
            barPoint=QtCore.QPoint(0, self.height-1.3*textHeight)
            w.scaleBarLabel.setPos(QtCore.QPointF(QtCore.QPointF(width_pixels/2-textWidth/2,self.height-textHeight)))
        if orientation=='horizontal':    
            w.scaleBarLabel.bar.setRect(QtCore.QRectF(barPoint, QtCore.QSizeF(width_pixels,textHeight/4)))
        elif orientation=='vertical':    
            w.scaleBarLabel.bar.setRect(QtCore.QRectF(barPoint, QtCore.QSizeF(textHeight/4,width_pixels)))

        if show_label == False:
            w.scaleBarLabel.hide()
        
    def preview(self):
        unit = self.getValue('unit')
        width_units=self.getValue('width_units')
        width_pixels=self.getValue('width_pixels')
        font_size=self.getValue('font_size')
        color=self.getValue('color')
        background=self.getValue('background')
        location=self.getValue('location')
        orientation=self.getValue('orientation')
        show=self.getValue('show')
        show_label=self.getValue('show_label')
        self.__call__(unit,width_units, width_pixels, font_size, color, background, location, orientation,show,show_label)

