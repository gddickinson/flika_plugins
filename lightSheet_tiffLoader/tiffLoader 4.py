import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
flika_version = flika.__version__
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import os
from .czifile import *

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

class Load_tiff ():
    """ load_tiff()
    This function loads tiff files from lightsheet experiments with multiple channels and volumes.

    """

    def __init__(self):
        pass

    def gui(self):
        filetypes = 'Image Files (*.tif *.tiff *.czi);;All Files (*.*)'
        prompt = 'Open File'
        filename = open_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None
        
        self.openTiff(filename)    
            
    def openTiff(self, filename):
        ext = os.path.splitext(filename)[-1]
        
        if ext in ['.tif', '.tiff', 'stk']:
            Tiff = tifffile.TiffFile(str(filename))

            A = Tiff.asarray()
            Tiff.close()
            
            axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]
            
        elif ext == '.czi':
            #import czifile
            czi = CziFile(filename)
            A = czi.asarray()
            czi.close()
            
            axes = [ax for ax in czi.axes]
            
            ##remove axes length ==1
            #get shape
            shape = A.shape
            #squeeze array to remove length ==1
            A = A.squeeze()
            #remove axis length 1 labels
            toRemove = []
            for n, i in enumerate(shape):
                if i == 1:
                    toRemove.append(n)
                    
            delete_multiple_element(axes,toRemove)
            
            #convert labels to tiff format
            for n, i in enumerate(axes):
                if i == 'T':
                    axes[n] = 'time'
                elif i == 'C':
                    axes[n] = 'channel'
                elif i == 'Y':
                    axes[n] = 'height'
                elif i == 'X':
                    axes[n] = 'width'
                elif i == 'Z':
                    axes[n] = 'depth'                    
          
            
            
        else:
            msg = "Could not open.  Filetype for '{}' not recognized".format(filename)
            g.alert(msg)
            if filename in g.settings['recent_files']:
                g.settings['recent_files'].remove(filename)
            # make_recent_menu()
            return
            
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

load_tiff = Load_tiff()

