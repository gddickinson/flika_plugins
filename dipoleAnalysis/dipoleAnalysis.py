from __future__ import division                 #to avoid integer division problem
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage.interpolation import shift
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika

from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
from flika import *
from flika.process.file_ import *
from flika.process.filters import *
from flika.process.overlay import *
from flika.process.binary import *
from flika.window import *
import os
from os.path import expanduser

from skimage.color import rgb2gray

from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu, threshold_adaptive
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import glob
from skimage.draw import ellipse
from skimage.transform import rotate
import math
from statistics import mean, median
from scipy import pi, dot, sin, cos
import pandas as pd
from skimage.morphology import watershed
from skimage import morphology
from skimage.morphology import disk
import scipy
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from flika import *
from flika.process.file_ import *
from flika.process.filters import *
from flika.window import *




flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


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
        #returns square of distance b/w two points
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
      
    #From Cosine law   
    alpha = np.arccos((b2 + c2 - a2)/(2*b*c)) 
    beta = np.arccos((a2 + c2 - b2)/(2*a*c)) 
    gamma = np.arccos((a2 + b2 - c2)/(2*a*b)) 
      
    # Converting to degrees 
    alpha = alpha * 180 / pi 
    beta = beta * 180 / pi 
    gamma = gamma * 180 / pi 
    
    return (alpha, beta, gamma)
######################################################################################



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
        |

    Returns:
        |Cropped clusters
        |Rotated clusters

    """
    def __init__(self):
        if g.settings['dipoleAnalysis'] is None:
            s = dict()
            s['resultsFolder1'] = None
            s['resultsFolder2'] = None
            s['dipoleWindow'] = None
            s['twoColorWindow'] = None             
            g.settings['dipoleAnalysis'] = s
                
        BaseProcess_noPriorWindow.__init__(self)

    def __call__(self,resultsFolder1,resultsFolder2,keepSourceWindow=False):
        '''
        
        '''
        g.settings['dipoleAnalysis']['resultsFolder1'] = resultsFolder1
        g.settings['dipoleAnalysis']['resultsFolder2'] = resultsFolder2

        g.m.statusBar().showMessage("Starting Dipole Analysis...")
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
                 

    def gui(self):
        s=g.settings['dipoleAnalysis']
        self.gui_reset()
        
        self.resultsFolder1 = FolderSelector('*.txt')
        self.resultsFolder2 = FolderSelector('*.txt')
        
        self.dipoleWindow = WindowSelector()
        self.twoColorWindow = WindowSelector()
        
        self.step1_Button = QPushButton('Extract Clusters from 2-Color Image')
        self.step1_Button.pressed.connect(self.step1)
        
        self.step2_Button = QPushButton('Rotate Clusters - Get Positions')
        self.step2_Button.pressed.connect(self.step2)
        
        self.step3_Button = QPushButton('Filter Clusters by Dipole Image')
        self.step3_Button.pressed.connect(self.step3)
        
        
        self.items.append({'name': 'resultsFolder1','string':'Results Folder 1 Location','object': self.resultsFolder1})     
        self.items.append({'name': 'resultsFolder2','string':'Results Folder 2 Location','object': self.resultsFolder2})   
        self.items.append({'name': 'dipoleWindow', 'string': 'Select Dipole Window', 'object': self.dipoleWindow})
        self.items.append({'name': 'twoColorWindow', 'string': 'Select two-color Window', 'object': self.twoColorWindow})
        self.items.append({'name': 'step1_Button', 'string': 'Step 1', 'object': self.step1_Button})
        self.items.append({'name': 'step2_Button', 'string': 'Step 2', 'object': self.step2_Button})
        self.items.append({'name': 'step3_Button', 'string': 'Step 3', 'object': self.step3_Button})      
        super().gui()

    def step1(self):
        #FIND CLUSTERS - CROP AND SAVE
        #superRes_crop_Path = r"C:\Users\georgedickinson\Documents\BSU_work\Brett - analysis for automation 2\tiffs\20190325 DMAT Dimer ED Unpaired 2Color.tif"
               
        #get cropped image
        superRes_crop = self.getValue('dipoleWindow')
        superRes_crop.setWindowTitle('superRes_crop')
        
        #convert superres to greyscale
        superResCrop_array = superRes_crop.imageview.getProcessedImage()
        superRes_grey = rgb2gray(superResCrop_array)
        
        Window(superRes_grey, 'SuperRes Grey')
        
        #blur image
        gaussian_blur(20,keepSourceWindow=False)
        
        #threshold
        threshold(0.02)        
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
        savePath1 = self.getValue('resultsFolder1')        
        np.savetxt(os.path.join(savePath1,'clusterInfo.csv'),exportList,delimiter=',',header='label,area,centeroid_X,centeroid_Y')
        
        #plot cluster boxes - crop and save images
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(superResCrop_array)
                
        i=1
        for region in measure.regionprops(labels):
            # take regions with large enough areas
            if region.area >= 2200:
        	
                # draw rectangle around segmented objects
                minr, minc, maxr, maxc = region.bbox
        
                #crop image and save tiff
                imageCrop = superResCrop_array[minr:maxr,minc:maxc,:]
                Window(imageCrop, str(i))
               
                (X,Y) = region.centroid 
                
                #save_file(r"C:\Users\georgedickinson\Documents\BSU_work\Brett - analysis for automation 2\tiffs\results\crop_{}_X{}_Y{}.tif".format(str(i), str(int(X)), str(int(Y))))
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

    def step2(self):
        #ROTATE CLUSTERS BLUE UP - IDENTIFY POSITIONS
        #get all cropped tiff paths
        #path = r"C:\Users\georgedickinson\Documents\BSU_work\Brett - analysis for automation 2\tiffs\results\*.tif"
        path = os.path.join(self.getValue('resultsFolder1'),'*.tif') 
        fileList = glob.glob(path)
        
        #savepath
        #savePath = r"C:\Users\georgedickinson\\Documents\BSU_work\Brett - analysis for automation 2\tiffs\results2"
        savePath = self.getValue('resultsFolder2') 
        
        #for testing
        #fileName = fileList[71]
        
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
                
                thresholded = threshold_adaptive(gray, 35)
                                
                distance_img = ndi.distance_transform_edt(thresholded)
                
                peaks_img = peak_local_max(distance_img, indices=False, min_distance=2)
                blurred_img = gaussian(peaks_img, 3)
                peaks_img = peak_local_max(blurred_img, indices=False)
                markers_img = measure.label(peaks_img)
                labelled_blobs = morphology.watershed(-distance_img, markers_img, mask=thresholded)
                num_blobs = len(np.unique(labelled_blobs))-1  # subtract 1 b/c background is labelled 0
                print ('number of blobs: %i' % num_blobs)
                
                
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
                
                
        #        #plot data
        #        fig, ax = plt.subplots()
        #        ax.imshow(np.swapaxes(file_img,0,1))
        #        
        #        for prop in blue_props:
        #            y0, x0 = prop.centroid
        #            ax.plot(y0, x0, '.g', markersize=15)
        #        
        #            minr, minc, maxr, maxc = prop.bbox
        #            bx = (minc, maxc, maxc, minc, minc)
        #            by = (minr, minr, maxr, maxr, minr)
        #            ax.plot(by, bx, '-b', linewidth=2.5)
        #        
        #        for prop in orange_props:
        #            y0, x0 = prop.centroid
        #            ax.plot(y0, x0, '.g', markersize=15)
        #        
        #            minr, minc, maxr, maxc = prop.bbox
        #            bx = (minc, maxc, maxc, minc, minc)
        #            by = (minr, minr, maxr, maxr, minr)
        #            ax.plot(by, bx, '-y', linewidth=2.5)        
        
        
                
                ######################################################################################
                
                ######## Analyse Orange up image ######################################################
                #showrotated image
                #Window(rotatedImg, 'rotated')
                #to avoid confusion - recalculate props for rotated image
                file_blur_rotated = gaussian(rotatedImg, 3)
                gray_rotated = rgb2gray(file_blur_rotated)
                
                thresholded_rotated = threshold_adaptive(gray_rotated, 35)        
                
                distance_img_rotated = ndi.distance_transform_edt(thresholded_rotated)
                
                #Window(distance_img_rotated, 'distance')
                
                peaks_img_rotated = peak_local_max(distance_img_rotated, indices=False, min_distance=2)
                #Window(peaks_img_rotated, 'peaks')
                blurred_img_rotated = gaussian(peaks_img_rotated, 3)
                peaks_img_rotated = peak_local_max(blurred_img_rotated, indices=False)
                markers_img_rotated = measure.label(peaks_img_rotated)
                labelled_blobs_rotated = morphology.watershed(-distance_img_rotated, markers_img_rotated, mask=thresholded_rotated)
                
                props_rotated = measure.regionprops(labelled_blobs_rotated)
                
                #Window(labelled_blobs_rotated, 'labelled blobs_rotated')
                
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
                    
                
                #show mask
                #Window(mask_1_rotated, 'mask 1 %s' % colourLabels[0])
                #Window(mask_2_rotated, 'mask 2 %s' % colourLabels[1])
                #Window(mask_3_rotated, 'mask 3 %s' % colourLabels[2])
                #Window(mask_4_rotated, 'mask 4 %s' % colourLabels[3])
                #Window(mask_5_rotated, 'mask 5 %s' % colourLabels[4])
                #Window(mask_6_rotated, 'mask 6 %s' % colourLabels[5])
                
                
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
                    allProps.append(['blue', prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                
                
                orangeCenteroids_rotated = []
                
                for prop in orange_props_rotated:
                    print('Orange-label: {} >> Object size: {}, Object position: {}'.format(prop.label, prop.area, prop.centroid))
                    orangeCenteroids_rotated.append(prop.centroid)
                    #exportList.append([prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                    allProps.append(['orange', prop.label, prop.area, prop.centroid[0],prop.centroid[1]])
                
                
                
                #plot data
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
                #create data frame for easier searching/filtering
                allPropsDF = pd.DataFrame(allProps,columns=['colour','label','area','centeroid-x','centeroid-y'])
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
        
               
                #line1
                line1_start_x = float(allPropsDF[(allPropsDF.position  == 1)]['centeroid-x'])
                line1_start_y = float(allPropsDF[(allPropsDF.position  == 1)]['centeroid-y'])
                
                line1_end_x = float(allPropsDF[(allPropsDF.position  == 4)]['centeroid-x'])
                line1_end_y = float(allPropsDF[(allPropsDF.position  == 4)]['centeroid-y'])
                
                ax.plot([line1_start_x,line1_end_x], [line1_start_y,line1_end_y], '-g')
                
                #line2
                line2_start_x = float(allPropsDF[(allPropsDF.position  == 6)]['centeroid-x'])
                line2_start_y = float(allPropsDF[(allPropsDF.position  == 6)]['centeroid-y'])
                
                line2_end_x = float(allPropsDF[(allPropsDF.position  == 2)]['centeroid-x'])
                line2_end_y = float(allPropsDF[(allPropsDF.position  == 2)]['centeroid-y'])
                
                ax.plot([line2_start_x,line2_end_x], [line2_start_y,line2_end_y], '-g')
                
                #line3
                line3_start_x = float(allPropsDF[(allPropsDF.position  == 5)]['centeroid-x'])
                line3_start_y = float(allPropsDF[(allPropsDF.position  == 5)]['centeroid-y'])
                
                line3_end_x = float(allPropsDF[(allPropsDF.position  == 3)]['centeroid-x'])
                line3_end_y = float(allPropsDF[(allPropsDF.position  == 3)]['centeroid-y'])
                
                ax.plot([line3_start_x,line3_end_x], [line3_start_y,line3_end_y], '-g')
                
                
                #get intersect point (using cramer's rule)
                line1 = line([line1_start_x,line1_start_y],[line1_end_x,line1_end_y]) 
                line2 = line([line2_start_x,line2_start_y],[line2_end_x,line2_end_y])                 
                line1_2_intersect = intersection(line1,line2)
                
                #get angle 1
                A = (line1_start_x,line1_start_y)
                B = (line2_start_x,line2_start_y)
                C = line1_2_intersect
                
                line1_2_angle = triangleAngles(A,B,C)[2]        
                                
                line1 = line([line1_start_x,line1_start_y],[line1_end_x,line1_end_y]) 
                line3 = line([line3_start_x,line3_start_y],[line3_end_x,line3_end_y])                 
                line1_3_intersect = intersection(line1,line3)               
                
                #get angle 2
                A = (line1_end_x,line1_end_y)
                B = (line3_end_x,line3_end_y)
                C = line1_3_intersect
                
                line1_3_angle = triangleAngles(A,B,C)[2]   
               
                fig.suptitle(fileName.split('\\')[-1])
                
                #ax.axis((0, 600, 600, 0))
                
                #uncomment to show plots
                #plt.show()
                
                fig.savefig(savePath + '\\' + str(fileNumber) + '_' + fileName.split('\\')[-1].split('.')[0] + '_Rot_' +str(int(rotation)) + '_Butter'+side + '_Angle1_' +str(int(line1_2_angle)) + '_Angle2_' +str(int(line1_3_angle)) )
                fileNumber += 1
                  
            except:
                print('error - skipping: ' + fileName.split('\\')[-1])
                fileNumber += 1
        return

    def step3(self):
        pass
        return

dipoleAnalysis = DipoleAnalysis()

