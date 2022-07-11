import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
flika_version = flika.__version__
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui

class Load_tiff ():
    """ load_tiff()
    This function loads tiff files from lightsheet experiments with multiple channels and volumes.

    """

    def __init__(self):
        pass

    def gui(self):
        filetypes = 'Image Files (*.tif *.tiff);;All Files (*.*)'
        prompt = 'Open File'
        self.filename = open_file_gui(prompt, filetypes=filetypes)
        if self.filename is None:
            return None
        
        self.openTiff(self.filename)    
            
    def openTiff(self, filename):
        Tiff = tifffile.TiffFile(str(filename))

        A = Tiff.asarray()
        Tiff.close()
        axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]
        print(axes)

        if set(axes) == set(['time', 'depth', 'height', 'width']):  # single channel, multi-volume
            target_axes = ['time', 'depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nScans, nFrames, x, y = A.shape

            #interleaved = np.zeros((nScans*nFrames,x,y))
            #
            #z = 0
            #for i in np.arange(nFrames):
            #    for j in np.arange(nScans):
            #        interleaved[z] = A[j%nScans][i] 
            #        z = z +1
            #newWindow = Window(interleaved,'Loaded Tiff')

            A = A.reshape(nScans*nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')
            
            
        elif set(axes) == set(['series', 'height', 'width']):  # single channel, single-volume
            target_axes = ['series', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')

        elif set(axes) == set(['depth', 'height', 'width']):  # single channel, single-volume
            target_axes = ['depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')                      
            
        elif set(axes) == set(['time', 'height', 'width']):  # single channel, single-volume
            target_axes = ['time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')
            
        elif set(axes) == set(['time', 'depth', 'channel', 'height', 'width']):  # multi-channel, multi-volume
            target_axes = ['channel','time','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            n1Scans, n1Frames, x1, y1 = B.shape
            n2Scans, n2Frames, x2, y2 = C.shape

            B = B.reshape(n1Scans*n1Frames,x1,y1)
            C = C.reshape(n2Scans*n2Frames,x2,y2)

            channel_1 = Window(B,'Channel 1')
            channel_2 = Window(C,'Channel 2')

            #clear A array to reduce memory use
            A = np.zeros((2,2))            
            
        elif set(axes) == set(['depth', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            channel_1 = Window(B,'Channel 1')
            channel_2 = Window(C,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            
        elif set(axes) == set(['time', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            channel_1 = Window(B,'Channel 1')
            channel_2 = Window(C,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))

    def getFileName(self):
        return self.filename

load_tiff = Load_tiff()

