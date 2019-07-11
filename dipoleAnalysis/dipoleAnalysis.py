from __future__ import division                 #to avoid integer division problem
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
#from scipy.ndimage.interpolation import shift
from flika.window import Window
import flika.global_vars as g
#import pyqtgraph as pg
#from time import time
from distutils.version import StrictVersion
import flika

#from flika.utils.io import tifffile
#from flika.process.file_ import get_permutation_tuple
#from flika.utils.misc import open_file_gui
from flika import *
from flika.process.file_ import *
from flika.process.filters import *
from flika.process.overlay import *
from flika.process.binary import *
from flika.window import *
import os
from os.path import expanduser

from skimage.color import rgb2gray

#from skimage.io import imread, imshow
from skimage.feature import peak_local_max

try:
    from skimage.filters import gaussian, threshold_local
    skimageVersion = 0
except:
    from skimage.filter import gaussian, threshold_adaptive
    skimageVersion = 1

from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import glob
#from skimage.draw import ellipse
#from skimage.transform import rotate
import math
from statistics import mean
from scipy import pi
import pandas as pd
#from skimage.morphology import watershed, disk
from skimage import morphology

import scipy
from scipy import ndimage as ndi

from flika import *
from flika.process.file_ import *
from flika.process.filters import *
from flika.window import *


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


#for testing
#start_flika()

###################     helper functions    ########################################
rad = lambda ang: ang*pi/180 

def angle2points(p1, p2):
    ang = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
    return np.rad2deg(ang)

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
    
def triangleAngles(A,B,C):    
    def lengthSquare(p1,p2):
        #distance squared
        xDiff = p1[0] - p2[0]
        yDiff = p1[1] - p2[1]
        return (xDiff*xDiff) + (yDiff*yDiff)   
        
    # Square of lengths a2, b2, c2 
    a2 = float(lengthSquare(B,C)) 
    b2 = float(lengthSquare(A,C)) 
    c2 = float(lengthSquare(A,B)) 
      
    #length of sides a, b, c 
    a = np.sqrt(a2) 
    b = np.sqrt(b2)
    c = np.sqrt(c2)
      
    #get angle From Cosine law   
    alpha = np.arccos((b2 + c2 - a2)/(2*b*c)) 
    beta = np.arccos((a2 + c2 - b2)/(2*a*c)) 
    gamma = np.arccos((a2 + b2 - c2)/(2*a*b)) 
      
    # Convert to degrees 
    alpha = alpha * 180 / pi 
    beta = beta * 180 / pi 
    gamma = gamma * 180 / pi 
    
    return (alpha, beta, gamma)
####################################################################################

class FolderSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
        self.button=QPushButton('Select Folder')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.folder = ''
        
    def buttonclicked(self):
        prompt = 'testing folderSelector'
        self.folder = QFileDialog.getExistingDirectory(g.m, "Select recording folder.", expanduser("~"), QFileDialog.ShowDirsOnly)
        self.label.setText('...'+os.path.split(self.folder)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])    
    
    
class DipoleAnalysis(BaseProcess_noPriorWindow):
    """
    Carry out dipole analysis 
    Input:
        |dipole Tiff
        |twoColor tiff

    Parameters:
        |Step 1: Guassian Blur
        |Step 1: Threshold
        
        |Step 2: Guassian Blur
        |Step 2: Threshold

        |Step 3: Guassian Blur
        |Step 3: Threshold
        |Step 3: Minimum number of blobs
        

    Returns:
        |Step 1: Two-color imaged masked by dipole image
        |Step 2: Cropped clusters from two-color image
        |Step 3: Rotated clusters from cropped clusters

    """
    def __init__(self):
        if g.settings['dipoleAnalysis'] is None or 'step2minClusterArea' not in g.settings['dipoleAnalysis'] :
            s = dict()
            s['resultsFolder1'] = None
            s['resultsFolder2'] = None
            s['resultsFolder3'] = None           
            s['dipoleWindow'] = None
            s['twoColorWindow'] = None  
            s['twoColorWindow_crop'] = None   
            s['step1Gaussian'] = False 
            s['step1Threshold'] = False    
            s['step1GuassianValue'] = 150
            s['step1ThresholdValue'] = 0.025
            s['step2GuassianValue'] = 20
            s['step2ThresholdValue'] = 0.02
            s['step2minClusterArea'] = 2200
            
            g.settings['dipoleAnalysis'] = s
                
        BaseProcess_noPriorWindow.__init__(self)

    def __call__(self,resultsFolder1,resultsFolder2,resultsFolder3,step1Gaussian,step1Threshold,step2Gaussian,step2Threshold,step2minClusterArea,keepSourceWindow=False):
        '''        
        '''
        g.settings['dipoleAnalysis']['resultsFolder1'] = resultsFolder1
        g.settings['dipoleAnalysis']['resultsFolder2'] = resultsFolder2
        g.settings['dipoleAnalysis']['resultsFolder3'] = resultsFolder3        
        g.settings['dipoleAnalysis']['step1Gaussian'] = step1Gaussian        
        g.settings['dipoleAnalysis']['step1Threshold'] = step1Threshold
        g.settings['dipoleAnalysis']['step2Gaussian'] = step2Gaussian        
        g.settings['dipoleAnalysis']['step2Threshold'] = step2Threshold 
        g.settings['dipoleAnalysis']['step2minClusterArea'] = step2minClusterArea         
        
        g.m.statusBar().showMessage("Starting Dipole Analysis...")
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
                 

    def gui(self):
        s=g.settings['dipoleAnalysis']
        self.gui_reset()
        #paths
        self.resultsFolder1 = FolderSelector('*.txt')
        self.resultsFolder2 = FolderSelector('*.txt')
        self.resultsFolder3 = FolderSelector('*.txt')        
        #windows
        self.dipoleWindow = WindowSelector()
        self.twoColorWindow = WindowSelector()
        self.twoColorWindow_crop = WindowSelector()
        #buttons
        self.step2_Button = QPushButton('Extract Clusters from 2-Color Image')
        self.step2_Button.pressed.connect(self.step2)        
        self.step3_Button = QPushButton('Rotate Clusters - Get Positions')
        self.step3_Button.pressed.connect(self.step3)        
        self.step1_Button = QPushButton('Filter Clusters by Dipole Image')
        self.step1_Button.pressed.connect(self.step1)        
        #checkboxes
        self.step1Gaussian = CheckBox()
        self.step1Threshold = CheckBox()        
        #Spinboxes
        self.step1GuassianValue = pg.SpinBox(int= True, step= 1)
        self.step1GuassianValue.setValue(s['step1GuassianValue'])
        self.step1ThresholdValue = pg.SpinBox(int= False, step= .001)
        self.step1ThresholdValue.setValue(s['step1ThresholdValue'])      
        
        self.step2GuassianValue = pg.SpinBox(int= True, step= 1)
        self.step2GuassianValue.setValue(s['step2GuassianValue'])
        self.step2ThresholdValue = pg.SpinBox(int= False, step= .001)
        self.step2ThresholdValue.setValue(s['step2ThresholdValue'])  
        
        self.step2minClusterArea  = pg.SpinBox(int= True, step= 1)
        self.step2minClusterArea.setValue(s['step2minClusterArea'])
        
        #populate GUI
        self.items.append({'name': 'resultsFolder1','string':'Results Folder 1 (for filtered two-color image)','object': self.resultsFolder1})            
        self.items.append({'name': 'resultsFolder2','string':'Results Folder 2 (for extracted clusters)','object': self.resultsFolder2})     
        self.items.append({'name': 'resultsFolder3','string':'Results Folder 3 (for final rotated clusters)','object': self.resultsFolder3})   
        self.items.append({'name': 'dipoleWindow', 'string': 'Dipole Window (Step 1)', 'object': self.dipoleWindow})
        self.items.append({'name': 'twoColorWindow', 'string': 'Two-color Window (Step 1)', 'object': self.twoColorWindow})
        self.items.append({'name': 'twoColorWindow_crop', 'string': 'Two-color Window for cluster analysis (Step 2)', 'object': self.twoColorWindow_crop})         
        self.items.append({'name': 'step1Gaussian', 'string': 'Display Step 1 Gaussian Image', 'object': self.step1Gaussian})  
        self.items.append({'name': 'step1Threshold', 'string': 'Display Step 1 Thresholded Image', 'object': self.step1Threshold})
        self.items.append({'name': 'step1GuassianValue', 'string': 'Step 1 Gaussian Value', 'object': self.step1GuassianValue})  
        self.items.append({'name': 'step1ThresholdValue', 'string': 'Step 1 Threshold Value', 'object': self.step1ThresholdValue})
        self.items.append({'name': 'step2GuassianValue', 'string': 'Step 2 Gaussian Value', 'object': self.step2GuassianValue})  
        self.items.append({'name': 'step2ThresholdValue', 'string': 'Step 2 Threshold Value', 'object': self.step2ThresholdValue})
        self.items.append({'name': 'step2minClusterArea', 'string': 'Step 2 Minimum Cluster Area', 'object': self.step2minClusterArea})       
          
        self.items.append({'name': 'step1_Button', 'string': 'Step 1', 'object': self.step1_Button})
        self.items.append({'name': 'step2_Button', 'string': 'Step 2', 'object': self.step2_Button})
        self.items.append({'name': 'step3_Button', 'string': 'Step 3', 'object': self.step3_Button})      
        super().gui()


    def step1(self):      
        #save path       
        savePath = self.getValue('resultsFolder1') 
        
        #get dipole data
        dipole = self.getValue('dipoleWindow')
        dipole.setWindowTitle('dipole')
                
        #gaussian blur
        gaussianBlurred = gaussian(dipole.imageview.getProcessedImage(), self.getValue('step1GuassianValue'))
        
        if self.getValue('step1Gaussian'):
            Window(gaussianBlurred, 'dipole - gaussian blurred')
        
        #threshold
        dipole_thresh = gaussianBlurred > self.getValue('step1ThresholdValue')
        dipole_thresh = dipole_thresh.astype(int)
        
        if self.getValue('step1Threshold'):       
            Window(dipole_thresh, 'dipole - thresholded')
                
        #get superResData
        superRes = self.getValue('twoColorWindow')
        superRes.setWindowTitle('twoColorSuperRes')
                
        #overlay
        #overlay = background(dipole_thresh,superRes,0.5, True)
                
        #mask by threshold
        superRes_crop = superRes.imageview.getProcessedImage()       
        mask = dipole_thresh < 1
        superRes_crop[mask] = [0,0,0] 
        
        #create window
        Window(superRes_crop, 'twoColor_Crop')
        
        #save image
        save_file(os.path.join(savePath,'twoColor_crop.tif'))
        return


    def step2(self):
        #FIND CLUSTERS - CROP AND SAVE
        #superRes_crop_Path = r"C:\Users\georgedickinson\Documents\BSU_work\Brett - analysis for automation 2\tiffs\20190325 DMAT Dimer ED Unpaired 2Color.tif"
               
        #set image windows
        superRes_crop = self.getValue('twoColorWindow_crop')
        superRes_crop.setWindowTitle('twoColorWindow_crop')
        
        #convert superres to greyscale
        superResCrop_array = superRes_crop.imageview.getProcessedImage()
        superRes_grey = rgb2gray(superResCrop_array)
        
        Window(superRes_grey, 'SuperRes Grey')
        
        #blur image
        gaussian_blur(self.getValue('step2GuassianValue'),keepSourceWindow=False)
        
        #threshold
        threshold(self.getValue('step2ThresholdValue'))        
        superRes_blur_thresh = g.m.currentWindow        
        superRes_blur_thresh_Array = superRes_blur_thresh.imageview.getProcessedImage()
        
        #blob detection
        labels = measure.label(superRes_blur_thresh_Array)
        
        exportList = []
        props = measure.regionprops(labels)
        for prop in props:
        	print('Label: {} >> Object size: {}, Object position: {}'.format(prop.label, prop.area, prop.centroid))
        	exportList.append([prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
        
        #np.savetxt(r"C:\Users\georgedickinson\Documents\BSU_work\Brett - analysis for automation 2\tiffs\results\clusterInfo.csv",exportList,delimiter=',',header='label,area,centeroid_X,centeroid_Y')

        #export cluster data
        savePath1 = self.getValue('resultsFolder2')        
        np.savetxt(os.path.join(savePath1,'clusterInfo.csv'),exportList,delimiter=',',header='label,area,centeroid_X,centeroid_Y')
        
        #plot cluster boxes - crop and save images
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(superResCrop_array)
                
        i=1
        for region in measure.regionprops(labels):
            # take regions with large enough areas
            if region.area >= self.getValue('step2minClusterArea') :
        	
                # draw rectangle around segmented objects
                minr, minc, maxr, maxc = region.bbox
        
                #crop image
                imageCrop = superResCrop_array[minr:maxr,minc:maxc,:]
                Window(imageCrop, str(i))
                
                #get centeroid
                (X,Y) = region.centroid 
                
                #save image
                save_file(os.path.join(savePath1,"crop_{}_X{}_Y{}.tif".format(str(i), str(int(X)), str(int(Y)))))   
                close()
        
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        												fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            i += 1
        
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        return

    def step3(self):
        #ROTATE CLUSTERS BLUE UP - IDENTIFY POSITIONS
        #get all cropped tiff paths

        path = os.path.join(self.getValue('resultsFolder2'),'*.tif') 
        fileList = glob.glob(path)
        
        #savepath
        savePath = self.getValue('resultsFolder3') 
        
        #for testing
        #fileName = r"C:\Users\georgedickinson\Desktop\test-Brett\results2\crop_54_X11118_Y2758.tif"
        #fileName = r"C:\Users\georgedickinson\Desktop\test-Brett\results2\crop_7_X3087_Y2122.tif"
        #fileName = fileList[71]

        #for export
        exportList2 = []
        
        fileNumber = 0
        
        for fileName in fileList:
            
            try:
                
                #open file
                file = open_file(fileName)
                file_img = file.imageview.getProcessedImage()
                close()
                
                
                
  
                ####### Rotate cropped image so orange up #############################################
                
                file_blur = gaussian(file_img, 3)
                gray = rgb2gray(file_blur)
                
                if skimageVersion == 1:
                    thresholded = threshold_adaptive(gray, 35)
                else:
                    adaptive_thresh = threshold_local(gray, 35)
                    thresholded = gray > adaptive_thresh
                                
                distance_img = ndi.distance_transform_edt(thresholded)
                
                peaks_img = peak_local_max(distance_img, indices=False, min_distance=2)
                blurred_img = gaussian(peaks_img, 3)
                peaks_img = peak_local_max(blurred_img, indices=False)
                markers_img = measure.label(peaks_img)
                labelled_blobs = morphology.watershed(-distance_img, markers_img, mask=thresholded)
                num_blobs = len(np.unique(labelled_blobs))-1  # subtract 1 b/c background is labelled 0
                print ('number of blobs: %i' % num_blobs)
                
                #skip image if not enough blobs detected               
                if num_blobs != 6:
                    print ('Incorrect number of blobs detected - skipping: ' + fileName.split('\\')[-1])
                    fileNumber += 1
                    continue
                
                props = measure.regionprops(labelled_blobs)
                                
                mask_1 = (np.copy(file_img))
                mask_1[labelled_blobs !=1] = 0
                
                mask_2 = (np.copy(file_img))
                mask_2[labelled_blobs !=2] = 0
                
                mask_3 = (np.copy(file_img))
                mask_3[labelled_blobs !=3] = 0
                
                mask_4 = (np.copy(file_img))
                mask_4[labelled_blobs !=4] = 0
                
                mask_5 = (np.copy(file_img))
                mask_5[labelled_blobs !=5] = 0
                
                mask_6 = (np.copy(file_img))
                mask_6[labelled_blobs !=6] = 0
                
                
                #determine if individual blobs blue or orange
                meanRGB_1 = (np.mean(mask_1[:, :, 0]), np.mean(mask_1[:, :, 1]), np.mean(mask_1[:, :, 2]))
                meanRGB_2 = (np.mean(mask_2[:, :, 0]), np.mean(mask_2[:, :, 1]), np.mean(mask_2[:, :, 2]))
                meanRGB_3 = (np.mean(mask_3[:, :, 0]), np.mean(mask_3[:, :, 1]), np.mean(mask_3[:, :, 2]))
                meanRGB_4 = (np.mean(mask_4[:, :, 0]), np.mean(mask_4[:, :, 1]), np.mean(mask_4[:, :, 2]))
                meanRGB_5 = (np.mean(mask_5[:, :, 0]), np.mean(mask_5[:, :, 1]), np.mean(mask_5[:, :, 2]))
                meanRGB_6 = (np.mean(mask_6[:, :, 0]), np.mean(mask_6[:, :, 1]), np.mean(mask_6[:, :, 2]))
                
                colourLabels=[]
                maskMeanList = [meanRGB_1,meanRGB_2,meanRGB_3,meanRGB_4,meanRGB_5,meanRGB_6]
                
                def returnLabel(meanRGB):
                    if meanRGB[0] > meanRGB[2]:
                        return 'orange'
                    else:
                        return 'blue'
                
                for meanRGB in maskMeanList:
                    colourLabels.append(returnLabel(meanRGB))
                                    
                
                blue_props = []
                orange_props = []
                
                blue_labels = []
                orange_labels = []
                
                for i in range(6):
                    if colourLabels[i] == 'blue':
                        blue_props.append(props[i])
                    else: 
                        orange_props.append(props[i])
                
                
                blueCenteroids = []
                
                for prop in blue_props:
                	print('Blue-label: {} >> Object size: {}, Object position: {}'.format(prop.label, prop.area, prop.centroid))
                	blueCenteroids.append(prop.centroid)
                	#exportList.append([prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                
                
                orangeCenteroids = []
                
                for prop in orange_props:
                	print('Orange-label: {} >> Object size: {}, Object position: {}'.format(prop.label, prop.area, prop.centroid))
                	orangeCenteroids.append(prop.centroid)
                	#exportList.append([prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                
                #rotate image so orange blobs above blue blobs
                #define line between middle of orange blob2 and middle of blue blobs
                blueCenteroids = np.array(blueCenteroids) 
                orangeCenteroids = np.array(orangeCenteroids) 
                
                blueCenteroids_xs = blueCenteroids[:,0]
                blueCenteroids_ys = blueCenteroids[:,1]
                
                orangeCenteroids_xs = orangeCenteroids[:,0]
                orangeCenteroids_ys = orangeCenteroids[:,1]  
                
                meanBlue_x = np.mean(blueCenteroids_xs)
                meanBlue_y = np.mean(blueCenteroids_ys)
                
                meanOrange_x = np.mean(orangeCenteroids_xs)
                meanOrange_y = np.mean(orangeCenteroids_ys)  
                
                #define center-point as mid-way between blue and orange blobs 
                centerPoint_x = mean([meanOrange_x,meanBlue_x])
                centerPoint_y = mean([meanOrange_y,meanBlue_y])        
                
                
                #get angle from center-point to blue edge   
                angle = angle2points((centerPoint_x,centerPoint_y), (meanOrange_x,meanOrange_y))
                       
                #rotate image so that Orange is on top (270 degrees)
                rotation = 270-angle
                rotatedImg = scipy.ndimage.rotate(file_img,rotation)
                
                     
                ######## Analyse Orange up image ######################################################
                #showrotated image
                #Window(rotatedImg, 'rotated')
                #to avoid confusion - recalculate props for rotated image
                file_blur_rotated = gaussian(rotatedImg, 3)
                gray_rotated = rgb2gray(file_blur_rotated)
                
                if skimageVersion == 1:
                    thresholded_rotated = threshold_adaptive(gray_rotated, 35) 
                else:
                    adaptive_thresh = threshold_local(gray_rotated, 35) 
                    thresholded_rotated = gray_rotated > adaptive_thresh
                
                distance_img_rotated = ndi.distance_transform_edt(thresholded_rotated)
                                
                peaks_img_rotated = peak_local_max(distance_img_rotated, indices=False, min_distance=2)

                blurred_img_rotated = gaussian(peaks_img_rotated, 3)
                peaks_img_rotated = peak_local_max(blurred_img_rotated, indices=False)
                markers_img_rotated = measure.label(peaks_img_rotated)
                labelled_blobs_rotated = morphology.watershed(-distance_img_rotated, markers_img_rotated, mask=thresholded_rotated)
                
                props_rotated = measure.regionprops(labelled_blobs_rotated)
                
                
                mask_1_rotated = (np.copy(rotatedImg))
                mask_1_rotated[labelled_blobs_rotated !=1] = 0
                
                mask_2_rotated = (np.copy(rotatedImg))
                mask_2_rotated[labelled_blobs_rotated !=2] = 0
                
                mask_3_rotated = (np.copy(rotatedImg))
                mask_3_rotated[labelled_blobs_rotated !=3] = 0
                
                mask_4_rotated = (np.copy(rotatedImg))
                mask_4_rotated[labelled_blobs_rotated !=4] = 0
                
                mask_5_rotated = (np.copy(rotatedImg))
                mask_5_rotated[labelled_blobs_rotated !=5] = 0
                
                mask_6_rotated = (np.copy(rotatedImg))
                mask_6_rotated[labelled_blobs_rotated !=6] = 0
                
                
                #determine if blobs blue or orange
                meanRGB_1_rotated = (np.mean(mask_1_rotated[:, :, 0]), np.mean(mask_1_rotated[:, :, 1]), np.mean(mask_1_rotated[:, :, 2]))
                meanRGB_2_rotated = (np.mean(mask_2_rotated[:, :, 0]), np.mean(mask_2_rotated[:, :, 1]), np.mean(mask_2_rotated[:, :, 2]))
                meanRGB_3_rotated = (np.mean(mask_3_rotated[:, :, 0]), np.mean(mask_3_rotated[:, :, 1]), np.mean(mask_3_rotated[:, :, 2]))
                meanRGB_4_rotated = (np.mean(mask_4_rotated[:, :, 0]), np.mean(mask_4_rotated[:, :, 1]), np.mean(mask_4_rotated[:, :, 2]))
                meanRGB_5_rotated = (np.mean(mask_5_rotated[:, :, 0]), np.mean(mask_5_rotated[:, :, 1]), np.mean(mask_5_rotated[:, :, 2]))
                meanRGB_6_rotated = (np.mean(mask_6_rotated[:, :, 0]), np.mean(mask_6_rotated[:, :, 1]), np.mean(mask_6_rotated[:, :, 2]))
                
                colourLabels_rotated=[]
                maskMeanList_rotated = [meanRGB_1_rotated,meanRGB_2_rotated,meanRGB_3_rotated,meanRGB_4_rotated,meanRGB_5_rotated,meanRGB_6_rotated]
                
                def returnLabel(meanRGB):
                    if meanRGB[0] > meanRGB[2]:
                        return 'orange'
                    else:
                        return 'blue'
                
                for meanRGB in maskMeanList_rotated:
                    colourLabels_rotated.append(returnLabel(meanRGB))
                                     
                
                blue_props_rotated = []
                orange_props_rotated = []
                
                blue_labels_rotated = []
                orange_labels_rotated = []
                
                for i in range(6):
                    if colourLabels_rotated[i] == 'blue':
                        blue_props_rotated.append(props_rotated[i])
                    else: 
                        orange_props_rotated.append(props_rotated[i])
                
                
                #create list to store all props
                allProps = []
                
                blueCenteroids_rotated = []
                
                for prop in blue_props_rotated:
                    print('Blue-label: {} >> Object size: {}, Object position: {}'.format(prop.label, prop.area, prop.centroid))
                    blueCenteroids_rotated.append(prop.centroid)
                    #exportList.append([prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                    allProps.append(['blue', prop.label, prop.area, prop.centroid[0],prop.centroid[1],prop.bbox])
                
                
                orangeCenteroids_rotated = []
                
                for prop in orange_props_rotated:
                    print('Orange-label: {} >> Object size: {}, Object position: {}'.format(prop.label, prop.area, prop.centroid))
                    orangeCenteroids_rotated.append(prop.centroid)
                    #exportList.append([prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                    allProps.append(['orange', prop.label, prop.area, prop.centroid[0],prop.centroid[1],prop.bbox])
                                
                ###############################
                ##### plot data ###############
                ###############################
                fig, ax = plt.subplots()
                ax.imshow(np.swapaxes(rotatedImg,0,1))
                
                for prop in blue_props_rotated:
                    y0, x0 = prop.centroid
                    ax.plot(y0, x0, '.g', markersize=15)
                
                    minr, minc, maxr, maxc = prop.bbox
                    bx = (minc, maxc, maxc, minc, minc)
                    by = (minr, minr, maxr, maxr, minr)
                    ax.plot(by, bx, '-b', linewidth=2.5)
                
                for prop in orange_props_rotated:
                    y0, x0 = prop.centroid
                    ax.plot(y0, x0, '.g', markersize=15)
                
                    minr, minc, maxr, maxc = prop.bbox
                    bx = (minc, maxc, maxc, minc, minc)
                    by = (minr, minr, maxr, maxr, minr)
                    ax.plot(by, bx, '-y', linewidth=2.5)
                
                #determine blob positions (1-6) for 'butter-side up' or 'butter-side down'
                #create pd dataframe for easier searching/filtering
                allPropsDF = pd.DataFrame(allProps,columns=['colour','label','area','centeroid-x','centeroid-y','bbox'])
                allPropsDF['position'] = 0
        
                #try butterside up layout
                side = 'UP'
                #p1
                allPropsDF.at[allPropsDF[(allPropsDF.colour == 'orange')]['centeroid-x'].argmin(), 'position'] = 1
                #p3
                allPropsDF.at[allPropsDF[(allPropsDF.colour == 'orange')]['centeroid-x'].argmax(), 'position'] = 3
                #p2       
                allPropsDF.loc[(allPropsDF['colour'] == 'orange') & (allPropsDF['position'] == 0), 'position'] = 2
                #p4
                allPropsDF.at[allPropsDF[(allPropsDF.colour == 'blue')]['centeroid-x'].argmax(), 'position'] = 4
                #p6
                allPropsDF.at[allPropsDF[(allPropsDF.colour == 'blue')]['centeroid-x'].argmin(), 'position'] = 6
                #p5             
                allPropsDF.loc[(allPropsDF['colour'] == 'blue') & (allPropsDF['position'] == 0), 'position'] = 5
        
        
        
                #test distance between 1-4 & 3-6, if 3-6 greater got to butterside down layout
                dist1_4 = math.hypot(float(allPropsDF[(allPropsDF.position  == 1)]['centeroid-x']) - float(allPropsDF[(allPropsDF.position  == 4)]['centeroid-x']), 
                                    float(allPropsDF[(allPropsDF.position  == 1)]['centeroid-y']) - float(allPropsDF[(allPropsDF.position  == 4)]['centeroid-y']))
                
                dist3_6 = math.hypot(float(allPropsDF[(allPropsDF.position  == 3)]['centeroid-x']) - float(allPropsDF[(allPropsDF.position  == 6)]['centeroid-x']), 
                                    float(allPropsDF[(allPropsDF.position  == 3)]['centeroid-y']) - float(allPropsDF[(allPropsDF.position  == 6)]['centeroid-y'])) 
        
                if dist3_6 > dist1_4:
                    side = 'DOWN'
                    #p1
                    allPropsDF.at[allPropsDF[(allPropsDF.colour == 'orange')]['centeroid-x'].argmin(), 'position'] = 3
                    #p3
                    allPropsDF.at[allPropsDF[(allPropsDF.colour == 'orange')]['centeroid-x'].argmax(), 'position'] = 1
                    #p2       
                    allPropsDF.loc[(allPropsDF['colour'] == 'orange') & (allPropsDF['position'] == 0), 'position'] = 2
                    #p4
                    allPropsDF.at[allPropsDF[(allPropsDF.colour == 'blue')]['centeroid-x'].argmax(), 'position'] = 6
                    #p6
                    allPropsDF.at[allPropsDF[(allPropsDF.colour == 'blue')]['centeroid-x'].argmin(), 'position'] = 4
                    #p5             
                    allPropsDF.loc[(allPropsDF['colour'] == 'blue') & (allPropsDF['position'] == 0), 'position'] = 5
        
               
                #line1 (Origami Angle)
                line1_start_x = float(allPropsDF[(allPropsDF.position  == 1)]['centeroid-x'])
                line1_start_y = float(allPropsDF[(allPropsDF.position  == 1)]['centeroid-y'])
                
                line1_end_x = float(allPropsDF[(allPropsDF.position  == 4)]['centeroid-x'])
                line1_end_y = float(allPropsDF[(allPropsDF.position  == 4)]['centeroid-y'])
                
                ax.plot([line1_start_x,line1_end_x], [line1_start_y,line1_end_y], '-g') #THIS IS THE ORIGAMI ANGLE LINE
                
                #get origami angle  
                if side == 'DOWN':
                    origami_angle = abs(angle2points((line1_end_x,line1_end_y), (line1_start_x,line1_start_y)))
                else:
                    origami_angle = 360 - angle2points((line1_start_x,line1_start_y), (line1_end_x,line1_end_y))

                #line1 bounds (Uncertainty of Origami Angle) 
                line1_start_bbox = allPropsDF[(allPropsDF.position  == 1)]['bbox'].values[0]
                line1_end_bbox = allPropsDF[(allPropsDF.position  == 4)]['bbox'].values[0] 
                
                if side == 'DOWN':                
                    line1_start_bounds1_x = float(line1_start_bbox[0])
                    line1_start_bounds1_y = float(line1_start_bbox[1])
                    line1_start_bounds2_x = float(line1_start_bbox[2])
                    line1_start_bounds2_y = float(line1_start_bbox[3])               
    
                    line1_end_bounds2_x = float(line1_end_bbox[0])
                    line1_end_bounds2_y = float(line1_end_bbox[1])
                    line1_end_bounds1_x = float(line1_end_bbox[2])
                    line1_end_bounds1_y = float(line1_end_bbox[3])  
                    
                else:
                    line1_start_bounds1_x = float(line1_start_bbox[2])
                    line1_start_bounds1_y = float(line1_start_bbox[1])
                    line1_start_bounds2_x = float(line1_start_bbox[0])
                    line1_start_bounds2_y = float(line1_start_bbox[3])               
    
                    line1_end_bounds2_x = float(line1_end_bbox[2])
                    line1_end_bounds2_y = float(line1_end_bbox[1])
                    line1_end_bounds1_x = float(line1_end_bbox[0])
                    line1_end_bounds1_y = float(line1_end_bbox[3])  

                ax.plot([line1_start_bounds1_x,line1_end_bounds1_x], [line1_start_bounds1_y,line1_end_bounds1_y], '-g') #ORIGAMI ANGLE BOUNDARYLINE 1
                ax.plot([line1_start_bounds2_x,line1_end_bounds2_x], [line1_start_bounds2_y,line1_end_bounds2_y], '-g') #ORIGAMI ANGLE BOUNDARYLINE 2
                
                #get origami boundary line angles   
                #origami_angle_bounds1 = angle2points((line1_start_bounds1_x,line1_start_bounds1_y), (line1_end_bounds1_x,line1_end_bounds1_y))
                #origami_angle_bounds2 = angle2points((line1_start_bounds2_x,line1_start_bounds2_y), (line1_end_bounds2_x,line1_end_bounds2_y))
                
                if side == 'DOWN':
                    origami_angle_bounds1 = abs( angle2points((line1_end_bounds1_x, line1_end_bounds1_y), (line1_start_bounds1_x, line1_start_bounds1_y)))
                    origami_angle_bounds2 = abs( angle2points((line1_end_bounds2_x, line1_end_bounds2_y), (line1_start_bounds2_x, line1_start_bounds2_y)))
                else:
                    origami_angle_bounds1 = 360 - angle2points((line1_start_bounds1_x, line1_start_bounds1_y), (line1_end_bounds1_x, line1_end_bounds1_y))
                    origami_angle_bounds2 = 360 - angle2points((line1_start_bounds2_x, line1_start_bounds2_y), (line1_end_bounds2_x, line1_end_bounds2_y))
                
                
                ##########
                
                #line2 (Helical Domain Angle)
                line2_start_x = float(allPropsDF[(allPropsDF.position  == 6)]['centeroid-x'])
                line2_start_y = float(allPropsDF[(allPropsDF.position  == 6)]['centeroid-y'])
                
                line2_end_x = float(allPropsDF[(allPropsDF.position  == 2)]['centeroid-x'])
                line2_end_y = float(allPropsDF[(allPropsDF.position  == 2)]['centeroid-y'])
                
                #ax.plot([line2_start_x,line2_end_x], [line2_start_y,line2_end_y], '-g')


                #line2 bounds
                line2_start_bbox = allPropsDF[(allPropsDF.position  == 6)]['bbox'].values[0]
                line2_end_bbox = allPropsDF[(allPropsDF.position  == 2)]['bbox'].values[0]
                
                if side == 'DOWN':                  
                    line2_start_bounds1_x = float(line2_start_bbox[2])
                    line2_start_bounds1_y = float(line2_start_bbox[1])
                    line2_start_bounds2_x = float(line2_start_bbox[0])
                    line2_start_bounds2_y = float(line2_start_bbox[3])                                  
                    line2_end_bounds2_x = float(line2_end_bbox[2])
                    line2_end_bounds2_y = float(line2_end_bbox[1])
                    line2_end_bounds1_x = float(line2_end_bbox[0])
                    line2_end_bounds1_y = float(line2_end_bbox[3])  
                    
                else:
                    line2_start_bounds1_x = float(line2_start_bbox[0])
                    line2_start_bounds1_y = float(line2_start_bbox[1])
                    line2_start_bounds2_x = float(line2_start_bbox[2])
                    line2_start_bounds2_y = float(line2_start_bbox[3])                                  
                    line2_end_bounds2_x = float(line2_end_bbox[0])
                    line2_end_bounds2_y = float(line2_end_bbox[1])
                    line2_end_bounds1_x = float(line2_end_bbox[2])
                    line2_end_bounds1_y = float(line2_end_bbox[3])                    
                

                #ax.plot([line2_start_bounds1_x,line2_end_bounds1_x], [line2_start_bounds1_y,line2_end_bounds1_y], '-g')
                ax.plot([line2_start_bounds2_x,line2_end_bounds2_x], [line2_start_bounds2_y,line2_end_bounds2_y], '-r') #THIS IS THE HELICAL ANGLE DOMAIN LINE (BUTTERSIDE DOWN)
                
                #get helical angle   (butterside down)
                helical_angle = abs(angle2points((line2_start_bounds2_x,line2_start_bounds2_y), (line2_end_bounds2_x,line2_end_bounds2_y)))


                #########
                
# =============================================================================
#                 #line3 (Helical Domain Angle for Butter Side Up)
#                 line3_start_x = float(allPropsDF[(allPropsDF.position  == 5)]['centeroid-x'])
#                 line3_start_y = float(allPropsDF[(allPropsDF.position  == 5)]['centeroid-y'])
#                 
#                 line3_end_x = float(allPropsDF[(allPropsDF.position  == 3)]['centeroid-x'])
#                 line3_end_y = float(allPropsDF[(allPropsDF.position  == 3)]['centeroid-y'])
#                 
#                 #ax.plot([line3_start_x,line3_end_x], [line3_start_y,line3_end_y], '-g')
#                 
#                 #line3 bounds 
#                 line3_start_bbox = allPropsDF[(allPropsDF.position  == 5)]['bbox'].values[0]
#                 line3_start_bounds1_x = float(line3_start_bbox[2])
#                 line3_start_bounds1_y = float(line3_start_bbox[1])
#                 line3_start_bounds2_x = float(line3_start_bbox[0])
#                 line3_start_bounds2_y = float(line3_start_bbox[3])               
# 
#                 line3_end_bbox = allPropsDF[(allPropsDF.position  == 3)]['bbox'].values[0]
#                 line3_end_bounds2_x = float(line3_end_bbox[2])
#                 line3_end_bounds2_y = float(line3_end_bbox[1])
#                 line3_end_bounds1_x = float(line3_end_bbox[0])
#                 line3_end_bounds1_y = float(line3_end_bbox[3])  
# 
#                 #ax.plot([line3_start_bounds1_x,line3_end_bounds1_x], [line3_start_bounds1_y,line3_end_bounds1_y], '-g')
#                 #ax.plot([line3_start_bounds2_x,line3_end_bounds2_x], [line3_start_bounds2_y,line3_end_bounds2_y], '-g')                
# =============================================================================
                
                                
                                
# =============================================================================
#                 #get intersect point (using cramer's rule)
#                 line1 = line([line1_start_x,line1_start_y],[line1_end_x,line1_end_y]) 
#                 line2 = line([line2_start_x,line2_start_y],[line2_end_x,line2_end_y])                 
#                 line1_2_intersect = intersection(line1,line2)
#                 
#                 #get angle 1
#                 A = (line1_start_x,line1_start_y)
#                 B = (line2_start_x,line2_start_y)
#                 C = line1_2_intersect
#                 
#                 line1_2_angles = triangleAngles(A,B,C)        
#                                 
#                 line1 = line([line1_start_x,line1_start_y],[line1_end_x,line1_end_y]) 
#                 line3 = line([line3_start_x,line3_start_y],[line3_end_x,line3_end_y])                 
#                 line1_3_intersect = intersection(line1,line3)               
#                 
#                 #get angle 2
#                 A = (line1_end_x,line1_end_y)
#                 B = (line3_end_x,line3_end_y)
#                 C = line1_3_intersect
#                 
#                 line1_3_angles = triangleAngles(A,B,C)   
# =============================================================================
               
                fig.suptitle(fileName.split('\\')[-1])
                
                #ax.axis((0, 600, 600, 0))
                
                #uncomment to show plots
                #plt.show()

                #convert rotation to clock-wise
                rotation = 360 - rotation
                
                exportList2.append([fileNumber,
                                    fileName.split('\\')[-1].split('.')[0],
                                    round(rotation,2),
                                    side,
                                    
                                    round(line1_start_x,2),
                                    round(line1_start_y,2),
                                    round(line1_end_x,2),
                                    round(line1_end_y,2),
                                    round(origami_angle,2),
                                                                  
                                    round(line1_start_bounds1_x,2),
                                    round(line1_start_bounds1_y,2),
                                    round(line1_end_bounds1_x,2),
                                    round(line1_end_bounds1_y,2),                                    
                                    round(origami_angle_bounds1,2),
                                    
                                    round(line1_start_bounds2_x,2),
                                    round(line1_start_bounds2_y,2),     
                                    round(line1_end_bounds2_x,2),
                                    round(line1_end_bounds2_y,2),  
                                    round(origami_angle_bounds2,2),
                                    
                                    round(line2_start_x,2),
                                    round(line2_start_y,2),
                                    round(line2_end_x,2),
                                    round(line2_end_y,2),
                                    round(helical_angle,2)])

                
                fig.savefig(savePath + '\\' + str(fileNumber) + '_' + fileName.split('\\')[-1].split('.')[0] +
                            '_Rot_' +str(int(rotation)) + '_Butter'+side + '_origamiAngle_' +str(int(origami_angle)) + '_helicalAngle_' +str(int(helical_angle)) )
                fileNumber += 1
                  
            except:
                print('error - skipping: ' + fileName.split('\\')[-1])
                fileNumber += 1
        
        #save datafile
        print(exportList2)
        np.savetxt(os.path.join(savePath,'finalInfo.csv'),
                   exportList2,delimiter=',',
                   header='fileNumber,fileName,rotationFromOriginal,side,line1_start_x,line1_start_y,line1_end_x,line1_end_y,origami_angle,line1_start_bounds1_x,line1_start_bounds1_y,line1_end_bounds1_x,line1_end_bounds1_y,origami_angle_bounds1,line1_start_bounds2_x,line1_start_bounds2_y,line1_end_bounds2_x,line1_end_bounds2_y,origami_angle_bounds2,line2_start_x,line2_start_y,line2_end_x,line2_end_y,helical_angle', 
                   fmt='%s')
        
        
        return


dipoleAnalysis = DipoleAnalysis()

