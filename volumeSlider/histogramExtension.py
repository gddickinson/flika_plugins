from pyqtgraph import HistogramLUTItem
from pyqtgraph import HistogramLUTWidget

class HistogramLUTItem_Overlay(HistogramLUTItem):
    '''
    GradientChanged overwritten to stop LUT updates from Histogram
    '''
    def __init__(self, **kwds): 
        
        HistogramLUTItem.__init__(self, **kwds)    
        self.overlay = True

    def gradientChanged(self):
        pass


class HistogramLUTWidget_Overlay(HistogramLUTWidget):
    '''
    '''
    def __init__(self, *args, **kargs): 
        
        HistogramLUTWidget.__init__(self, *args, **kargs)

        self.item = HistogramLUTItem_Overlay(*args, **kargs)
        self.setCentralItem(self.item)
        self.setMinimumWidth(95)

        
    def setLUT(self,lut):
        self.item.lut = lut