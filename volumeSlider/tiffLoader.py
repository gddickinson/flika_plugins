import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
from distutils.version import StrictVersion
from flika.window import Window

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

#flika.start_flika()

def openTiff(filename):
    Tiff = tifffile.TiffFile(str(filename))

    A = Tiff.asarray()
    B = []
    C = []
    Tiff.close()
    axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]
    print(axes)

    if set(axes) == set(['time', 'depth', 'height', 'width']):  # single channel, multi-volume
        target_axes = ['time', 'depth', 'width', 'height']
        perm = get_permutation_tuple(axes, target_axes)
        A = np.transpose(A, perm)
        nScans, nFrames, x, y = A.shape
        A = A.reshape(nScans*nFrames,x,y)
        #newWindow = Window(A,'Loaded Tiff')
                
    elif set(axes) == set(['series', 'height', 'width']):  # single channel, single-volume
        target_axes = ['series', 'width', 'height']
        perm = get_permutation_tuple(axes, target_axes)
        A = np.transpose(A, perm)
        nFrames, x, y = A.shape
        A = A.reshape(nFrames,x,y)
        #newWindow = Window(A,'Loaded Tiff')
        
    elif set(axes) == set(['time', 'height', 'width']):  # single channel, single-volume
        target_axes = ['time', 'width', 'height']
        perm = get_permutation_tuple(axes, target_axes)
        A = np.transpose(A, perm)
        nFrames, x, y = A.shape
        A = A.reshape(nFrames,x,y)
        #newWindow = Window(A,'Loaded Tiff')
        
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

        #channel_1 = Window(B,'Channel 1')
        #channel_2 = Window(C,'Channel 2')
           
        
    elif set(axes) == set(['depth', 'channel', 'height', 'width']):  # multi-channel, single volume
        target_axes = ['channel','depth', 'width', 'height']
        perm = get_permutation_tuple(axes, target_axes)
        A = np.transpose(A, perm)
        B = A[0]
        C = A[1]

        #channel_1 = Window(B,'Channel 1')
        #channel_2 = Window(C,'Channel 2')

        
    elif set(axes) == set(['time', 'channel', 'height', 'width']):  # multi-channel, single volume
        target_axes = ['channel','time', 'width', 'height']
        perm = get_permutation_tuple(axes, target_axes)
        A = np.transpose(A, perm)
        B = A[0]
        C = A[1]

        #channel_1 = Window(B,'Channel 1')
        #channel_2 = Window(C,'Channel 2')
        
    return A, B, C



#A, _, _ = openTiff(fileName)

#newWindow = Window(A,'Loaded Tiff') 



    