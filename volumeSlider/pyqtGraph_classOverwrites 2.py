import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph import mkPen
import pyqtgraph.opengl as gl
from OpenGL.GL import *



###############  New GLGraphicsItem class definition ################################
class GLBorderItem(gl.GLAxisItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
    Overwrite of GLAxisItem
    Displays borders of plot data

    """

    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.update()


    def size(self):
        return self.__size[:]


    def paint(self):

        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable( GL_BLEND )
        #glEnable( GL_ALPHA_TEST )
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin( GL_LINES )

        x,y,z = self.size()

        def zFrame(x,y,z,r=1,g=1,b=1,thickness=0.6):
            glColor4f(r, g, b, thickness)  # z
            glVertex3f(-(int(x)), -(int(y/2)), -(int(z/2)))
            glVertex3f(-(int(x)), -(int(y/2)), z-(int(z/2)))

            glColor4f(r, g, b, thickness)  # z
            glVertex3f(-(int(x)), y -(int(y/2)), -(int(z/2)))
            glVertex3f(-(int(x)), y -(int(y/2)), z-(int(z/2)))

            glColor4f(r, g, b, thickness)  # y
            glVertex3f(-(int(x)), -(int(y/2)), -(int(z/2)))
            glVertex3f(-(int(x)), y -(int(y/2)), -(int(z/2)))

            glColor4f(r, g, b, thickness)  # y
            glVertex3f(-(int(x)), -(int(y/2)), z-(int(z/2)))
            glVertex3f(-(int(x)), y -(int(y/2)), z-(int(z/2)))


        def xFrame(x,y,z,r=1,g=1,b=1,thickness=0.6):
            glColor4f(r, g, b, thickness)  # x is blue
            glVertex3f(x-(int(x/2)), -(int(y)), -(int(z/2)))
            glVertex3f((int(x/2))-x, -(int(y)), -(int(z/2)))

            glColor4f(r, g, b, thickness)  # x is blue
            glVertex3f(x-(int(x/2)), -(int(y)), z-(int(z/2)))
            glVertex3f((int(x/2))-x, -(int(y)), z-(int(z/2)))

            glColor4f(r, g, b, thickness)  # z
            glVertex3f(x-(int(x/2)), -(int(y)), -(int(z/2)))
            glVertex3f(x-(int(x/2)), -(int(y)), z-(int(z/2)))

            glColor4f(r, g, b, thickness)  # z
            glVertex3f((int(x/2))-x, -(int(y)), -(int(z/2)))
            glVertex3f((int(x/2))-x, -(int(y)), z-(int(z/2)))


        def box(x,y,z,r=1,g=1,b=1,thickness=0.6):
            zFrame(x/2,y,z,r=r,g=g,b=b,thickness=thickness)
            zFrame(x/2-x,y,z,r=r,g=g,b=b,thickness=thickness)
            xFrame(x,y/2,z,r=r,g=g,b=b,thickness=thickness)
            xFrame(x,-y/2,z,r=r,g=g,b=b,thickness=thickness)


        box(x,y,z)

        glEnd()


