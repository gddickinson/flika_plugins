# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:29:16 2020

@author: g_dic
"""

import numpy as np
from matplotlib import pyplot as plt
import re
import h5py
from skimage.transform import rescale
import flika
from flika import global_vars as g
import pandas as pd

headerList = ['Channel Name','X','Y','Xc','Yc','Height','Area','Width','Phi','Ax','BG','I','Frame','Length','Link','Valid','Z','Zc','Photons','Lateral Localization Accuracy','Xw','Yw','Xwc','Ywc','Zw','Zwc']
dataDict = {'Channel Name':['Fake_1'],
            'Height':[500],
            'Area':[500],
            'Width':[500],
            'Phi':[1.0],
            'Ax':[1.0],
            'BG':[1000],
            'I':[1000],
            'Frame':[1],
            'Length':[1],
            'Link':[1],
            'Valid':[1],
            'Photons':[10000],
            'Lateral Localization Accuracy':[10],
}

def saveFakeSuperRes(x,y,z, savePath):
    n_rows = len(x)        
    fake_DF = pd.DataFrame(columns=headerList)

    for name,value in dataDict.items():
        fake_DF[name] = value * n_rows

    fake_DF['X'] = x    
    fake_DF['Xc'] = x
    fake_DF['Xw'] = x    
    fake_DF['Xwc'] = x  
     
    fake_DF['Y'] = y    
    fake_DF['Yc'] = y
    fake_DF['Yw'] = y 
    fake_DF['Ywc'] = y     
    
    fake_DF['Z'] = z    
    fake_DF['Zc'] = z
    fake_DF['Zw'] = z    
    fake_DF['Zwc'] = z   
    
    #add empty channel (duplicate last row)
    emptyRow = list(fake_DF.iloc[-1])
    emptyRow[0] = 'Fake_2'
    a_series = pd. Series(emptyRow, index = fake_DF.columns)
    fake_DF = fake_DF.append(a_series, ignore_index=True)

    fake_DF.to_csv(savePath, index=None, sep='\t')
    print('Fake SuperRes file saved to :', savePath)
    return
