# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:38:09 2019

@author: George
"""

import os, time
import datetime
from os.path import expanduser
import numpy as np
from numpy import moveaxis
from scipy.ndimage.interpolation import zoom
from time import time
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.images import image_path
from skimage.transform import rescale
from flika.process.file_ import get_permutation_tuple
import sys
from time import sleep
import platform
import ast

#Get parameters as args

nSteps = int(sys.argv[1])
shift_factor = float(sys.argv[2])
theta = int(sys.argv[3])
triangle_scan = ast.literal_eval(sys.argv[4])
interpolate = ast.literal_eval(sys.argv[5])
trim_last_frame = ast.literal_eval(sys.argv[6])
zscan = ast.literal_eval(sys.argv[7])
nChannels = int(sys.argv[8])
volume = sys.argv[9]
recordingFolder = sys.argv[10]
exportFolder = sys.argv[11]

#if passing from terminal
#recordingFolder = "%r" % sys.argv[9]
#exportFolder = "%r" % sys.argv[10]

#print (recordingFolder)
#print (exportFolder)
#print (nSteps)
#print (shift_factor)
#print (theta)
#print (triangle_scan)
#print (interpolate)
#print (trim_last_frame)
#print (zscan)
#print (nChannels)

#testing parameters
#recordingFolder = "C:\\Users\\George\\Desktop\\testRun\\george_1color"
#exportFolder = "C:\\Users\\George\\Desktop\\testRun\\results2"  
#nSteps = 200
#shift_factor = 1
#theta = 45
#triangle_scan = False
#interpolate = False
#trim_last_frame = False
#zscan = False
#nChannels = 1
#volume = "0"


class VolumeAnalyzer():

    def __init__(self):
        pass

    def get_transformation_matrix(self,theta=45):
        """
        theta is the angle of the light sheet
        Look at the pdf in this folder.
        """
    
        theta = theta/360 * 2 * np.pi # in radians
        hx = np.cos(theta)
        sy = np.sin(theta)
     
        S = np.array([[1, hx, 0],
                      [0, sy, 0],
                      [0, 0, 1]])
        return S


    def get_transformation_coordinates(self, I, theta):
        negative_new_max = False
        S = self.get_transformation_matrix(theta)
        S_inv = np.linalg.inv(S)
        mx, my = I.shape
    
        four_corners = np.matmul(S, np.array([[0, 0, mx, mx],
                                              [0, my, 0, my],
                                              [1, 1, 1, 1]]))[:-1,:]
        range_x = np.round(np.array([np.min(four_corners[0]), np.max(four_corners[0])])).astype(np.int)
        range_y = np.round(np.array([np.min(four_corners[1]), np.max(four_corners[1])])).astype(np.int)
        all_new_coords = np.meshgrid(np.arange(range_x[0], range_x[1]), np.arange(range_y[0], range_y[1]))
        new_coords = [all_new_coords[0].flatten(), all_new_coords[1].flatten()]
        new_homog_coords = np.stack([new_coords[0], new_coords[1], np.ones(len(new_coords[0]))])
        old_coords = np.matmul(S_inv, new_homog_coords)
        old_coords = old_coords[:-1, :]
        old_coords = old_coords
        old_coords[0, old_coords[0, :] >= mx-1] = -1
        old_coords[1, old_coords[1, :] >= my-1] = -1
        old_coords[0, old_coords[0, :] < 1] = -1
        old_coords[1, old_coords[1, :] < 1] = -1
        new_coords[0] -= np.min(new_coords[0])
        keep_coords = np.logical_not(np.logical_or(old_coords[0] == -1, old_coords[1] == -1))
        new_coords = [new_coords[0][keep_coords], new_coords[1][keep_coords]]
        old_coords = [old_coords[0][keep_coords], old_coords[1][keep_coords]]
        return old_coords, new_coords
    
    def perform_shear_transform(self, A, shift_factor, interpolate, datatype, theta):
        A = moveaxis(A, [1, 3, 2, 0], [0, 1, 2, 3])
        m1, m2, m3, m4 = A.shape
        if interpolate:
            A_rescaled = np.zeros((m1*int(shift_factor), m2, m3, m4))
            for v in np.arange(m4):
                print('Upsampling Volume #{}/{}'.format(v+1, m4))
                A_rescaled[:, :, :, v] = rescale(A[:, :, :, v], (shift_factor, 1.), mode='constant', preserve_range=True)
        else:
            A_rescaled = np.repeat(A, shift_factor, axis=0)
        mx, my, mz, mt = A_rescaled.shape
        I = A_rescaled[:, :, 0, 0]
        old_coords, new_coords = self.get_transformation_coordinates(I, theta)
        old_coords = np.round(old_coords).astype(np.int)
        new_mx, new_my = np.max(new_coords[0]) + 1, np.max(new_coords[1]) + 1
        # I_transformed = np.zeros((new_mx, new_my))
        # I_transformed[new_coords[0], new_coords[1]] = I[old_coords[0], old_coords[1]]
        # Window(I_transformed)
        D = np.zeros((new_mx, new_my, mz, mt))
        D[new_coords[0], new_coords[1], :, :] = A_rescaled[old_coords[0], old_coords[1], :, :]
        E = moveaxis(D, [0, 1, 2, 3], [3, 1, 2, 0])
        E = np.flip(E, 1)
        #Window(E[0, :, :, :])
        E = E.astype(datatype)
        return E


    def analyse1Volume(self,
                        filename,
                        exportFolder = r"C:\Users\George\Desktop\testRun\results2",  
                        nSteps = 1,
                        shift_factor = 1,
                        theta = 45,
                        triangle_scan = False,
                        interpolate = False,
                        trim_last_frame = False,
                        zscan = False,
                        nChannels = 1):
        
        #load volume
        array_channel_1, array_channel_2 = load_tiff.openTiff(filename)
        #prepare volume
        A = np.copy(array_channel_1)
        #print(A.shape)
        
        #Preprocess array
        if zscan:
            A = A.swapaxes(1,2)
        
        mt, mx, my = A.shape
        
        if triangle_scan:
            for i in np.arange(mt // (nSteps * 2)):
                t0 = i * nSteps * 2 + nSteps
                tf = (i + 1) * nSteps * 2
                A[t0:tf] = A[tf:t0:-1]
        
        mv = mt // nSteps  # number of volumes
        
        A = A[:mv * nSteps]
        B = np.reshape(A, (mv, nSteps, mx, my))
        A_dataType = A.dtype
        
        A = np.zeros((2,2)) #clear array to save memory
        
        if trim_last_frame:
            B = B[:, :-1, :, :]
        
        D = self.perform_shear_transform(B, shift_factor, interpolate, A_dataType, theta)
        
        B = np.zeros((2,2)) #clear array to save memory
        
        #send volume to analyzer
        self.run(D,exportFolder)
        return

    def run(self, volume, exportPath): 
        #get variables
        self.volume = volume
        self.vol_shape = self.volume.shape
        mv,mz,mx,my= self.volume.shape
        self.currentAxisOrder=[0,1,2,3]
        self.current_v_Index=0
        self.current_z_Index=0
        self.current_x_Index=0
        self.current_y_Index=0
        self.exportPath = exportPath
        
        #run
        self.export_volume()
        
        return
               
    def getXview(self):
        vol = self.volume
        #assert currentAxisOrder==[0,1,2,3]
        vol=vol.swapaxes(1,2)
        #currentAxisOrder=[0,2,1,3]
        vol=vol.swapaxes(2,3)
        #currentAxisOrder=[0,2,3,1]  
        return vol

    def getYview(self):
        vol = self.volume
        #assert self.currentAxisOrder==[0,1,2,3]    
        vol=vol.swapaxes(1,3)
        #self.currentAxisOrder=[0,3,2,1]
        return vol
        
    def make_maxintensity(self):
        vol=self.volume
        new_vol=np.max(vol,1)
        #Window(new_vol, name=name)
        
    def export_volume(self):
        operations = [(self.volume,'top'), (self.getXview(),'Xview'), (self.getYview(),'Yview')]
    
        for operation in operations:
        
            vol = operation[0]
            folderName = operation[1]
            
            #if platform.system() == 'Windows':
            #    correctPath = self.exportPath.replace('/','\\')
            
            export_path = os.path.join(self.exportPath, folderName,'light_sheet_vols')
            #export_path = self.exportPath + '/' + folderName + '/' + 'light_sheet_vols'
            i=0
            while os.path.isdir(export_path+'_'+str(i)):
                i+=1
            export_path=export_path+'_'+str(i)
            os.makedirs(export_path, exist_ok=True) 
            for v in np.arange(len(vol)):
                A=vol[v]
                filename=os.path.join(export_path,str(v)+'.tiff')
                if len(A.shape)==3:
                    A=np.transpose(A,(0,2,1)) # This keeps the x and the y the same as in FIJI
                elif len(A.shape)==2:
                    A=np.transpose(A,(1,0))
                tifffile.imsave(filename, A)

        return

volumeAnalyzer = VolumeAnalyzer()  

class Load_tiff ():
    """ load_tiff()
    This function loads tiff files from lightsheet experiments with multiple channels and volumes.

    """

    def __init__(self):
        pass
  
            
    def openTiff(self, filename):
        Tiff = tifffile.TiffFile(str(filename))
        A = Tiff.asarray()
        Tiff.close()
        axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]

        if set(axes) == set(['time', 'depth', 'height', 'width']):  # single channel, multi-volume
            target_axes = ['time', 'depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nScans, nFrames, x, y = A.shape

            A = A.reshape(nScans*nFrames,x,y)
            #newWindow = Window(A,'Loaded Tiff')
            return A, None
            
        elif set(axes) == set(['series', 'height', 'width']):  # single channel, single-volume
            target_axes = ['series', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            #newWindow = Window(A,'Loaded Tiff')
            return A, None
            
        elif set(axes) == set(['time', 'height', 'width']):  # single channel, single-volume
            target_axes = ['time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            #newWindow = Window(A,'Loaded Tiff')
            return A, None
            
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
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return B, C
            
        elif set(axes) == set(['depth', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            #channel_1 = Window(B,'Channel 1')
            #channel_2 = Window(C,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return B, C
        
        elif set(axes) == set(['time', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]
            
            nChannels, nFrames, x, y = A.shape
            
            n1Frames, x1, y1 = B.shape
            n2Frames, x2, y2 = C.shape

            #uncomment to display original
            #self.channel_1 = Window(B,'Channel 1')
            #self.channel_2 = Window(B,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return B, C
        
load_tiff = Load_tiff() 



#############################
##Run analyzer for volume file received

volumeFolder = 'vol_' + volume
tifName = volumeFolder + '.tif'
filename = os.path.join(recordingFolder,volumeFolder,tifName)


volumeAnalyzer.analyse1Volume(
        
                        filename = filename,
                        exportFolder = exportFolder,  
                        nSteps = nSteps,
                        shift_factor = shift_factor,
                        theta = theta,
                        triangle_scan = triangle_scan,
                        interpolate = interpolate,
                        trim_last_frame = trim_last_frame,
                        zscan = zscan,
                        nChannels = nChannels
                        
                        )