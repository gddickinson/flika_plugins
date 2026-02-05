import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
w = gl.GLViewWidget()

# Antialias
pg.setConfigOptions(antialias=True)

w.show()
w.setWindowTitle('GLScatterPlotItem')

g = gl.GLGridItem()
w.addItem(g)

pos = np.random.randn(50, 3)*10
color = np.random.randn(50, 3)*10

sp1 = gl.GLScatterPlotItem(pos=pos, color=color, size=25)
w.addItem(sp1)

# Planes
gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
#gx.translate(-10, 0, 0)
w.addItem(gx)
gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
#gy.translate(0, -10, 0)
w.addItem(gy)
gz = gl.GLGridItem()
gz.scale(5.0/10.0, 5.0/10.0, 1.0/10.0)
#gz.translate(0, 0, -10)
w.addItem(gz)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()