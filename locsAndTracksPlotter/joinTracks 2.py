#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:25:24 2022

@author: george
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from scipy import stats, spatial
from pathlib import Path
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

#sys.setrecursionlimit(10000)
import os
cwd = os.path.dirname(os.path.abspath(__file__))

train_data_path = os.path.join(cwd,'tdTomato_37Degree_CytoD_training_feats.csv')

class JoinTracks():
    def __init__(self):
        #minimum number of link segments
        self.minLinkSegments = 4        

    
    def RadiusGyrationAsymmetrySkewnessKurtosis(self, trackDF):
        ''' Radius of Gyration and Asymmetry'''
        # Drop any skipped frames and convert trackDF to XY array
        points_array = np.array(trackDF[['x', 'y']].dropna())  
        # get Rg etc using Vivek's codes
        center = points_array.mean(0)
        normed_points = points_array - center[None, :]
        radiusGyration_tensor = np.einsum('im,in->mn', normed_points, normed_points)/len(points_array)
        eig_values, eig_vectors = np.linalg.eig(radiusGyration_tensor)
        radius_gyration_value = np.sqrt(np.sum(eig_values))
        asymmetry_numerator = pow((eig_values[0] - eig_values[1]), 2)
        asymmetry_denominator = 2 * (pow((eig_values[0] + eig_values[1]), 2))
        asymmetry_value = - math.log(1 - (asymmetry_numerator / asymmetry_denominator))
        maxcol = list(eig_values).index(max(eig_values))
        dominant_eig_vect = eig_vectors[:, maxcol]
        points_a = points_array[:-1]
        points_b = points_array[1:]
        ba = points_b - points_a
        proj_ba_dom_eig_vect = np.dot(ba, dominant_eig_vect) / np.power(np.linalg.norm(dominant_eig_vect), 2)
        skewness_value = stats.skew(proj_ba_dom_eig_vect)
        kurtosis_value = stats.kurtosis(proj_ba_dom_eig_vect)
        return radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value    
    
    # Fractal Dimension
    def FractalDimension(self, points_array):
        ####Vivek's code    
        #Check if points are on the same line:
        x0, y0 = points_array[0]
        points = [ (x, y) for x, y in points_array if ( (x != x0) or (y != y0) ) ]
        slopes = [ ((y - y0) / (x - x0)) if (x != x0) else None for x, y in points ]
        if all( s == slopes[0] for s in slopes):
            raise ValueError("Fractal Dimension cannot be calculated for points that are collinear")
        total_path_length = np.sum(pow(np.sum(pow(points_array[1:, :] - points_array[:-1, :], 2), axis=1), 0.5))
        stepCount = len(points_array)
        candidates = points_array[spatial.ConvexHull(points_array).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        maxIndex = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        largestDistance = dist_mat[maxIndex]
        fractal_dimension_value = math.log(stepCount) / math.log(stepCount * largestDistance * math.pow(total_path_length, -1))
        return fractal_dimension_value
    
    # Net Displacement
    def NetDisplacementEfficiency(self, points_array):      
        ####Vivek's code   
        net_displacement_value = np.linalg.norm(points_array[0]-points_array[-1])
        netDispSquared = pow(net_displacement_value, 2)
        points_a = points_array[1:, :]
        points_b = points_array[:-1, :]
        dist_ab_SumSquared = sum(pow(np.linalg.norm(points_a-points_b, axis=1), 2))
        efficiency_value = netDispSquared / ((len(points_array)-1) * dist_ab_SumSquared)
        return net_displacement_value, efficiency_value
    
    
    # Bending & Straightness Features
    def SummedSinesCosines(self, points_array):
        ## Vivek's code
        # Look for repeated positions in consecutive frames
        compare_against = points_array[:-1]
        # make a truth table identifying duplicates
        duplicates_table = points_array[1:] == compare_against
        # Sum the truth table across the rows, True = 1, False = 0
        duplicates_table = duplicates_table.sum(axis=1)
        # If both x and y are duplicates, value will be 2 (True + True == 2)
        duplicate_indices = np.where(duplicates_table == 2)
        # Remove the consecutive duplicates before sin, cos calc
        points_array = np.delete(points_array, duplicate_indices, axis=0)
        # Generate three sets of points
        points_set_a = points_array[:-2]
        points_set_b = points_array[1:-1]
        points_set_c = points_array[2:]
        # Generate two sets of vectors
        ab = points_set_b - points_set_a
        bc = points_set_c - points_set_b
        # Evaluate sin and cos values
        cross_products = np.cross(ab, bc)
        dot_products = np.einsum('ij,ij->i', ab, bc)
        product_magnitudes_ab_bc = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
        cos_vals = dot_products / product_magnitudes_ab_bc
        cos_mean_val = np.mean(cos_vals)
        sin_vals = cross_products / product_magnitudes_ab_bc
        sin_mean_val = np.mean(sin_vals)
        return sin_mean_val, sin_vals, cos_mean_val, cos_vals
    
    def getRadiusGyrationForAllTracksinDF(self, tracksDF):
        tracksToTest = tracksDF['track_number'].tolist()
        idTested = []
        radius_gyration_list=[] 
        asymmetry_list=[] 
        skewness_list=[] 
        kurtosis_list=[]  
        trackIntensity_mean = []
        trackIntensity_std = []
        
        for i in range(len(tracksToTest)):
            idToTest = tracksToTest[i]
            if idToTest not in idTested:
                radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value = self.RadiusGyrationAsymmetrySkewnessKurtosis(tracksDF[tracksDF['track_number']==idToTest])
                idTested.append(idToTest)
                #print(radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value)
                print('\r' + 'RG analysis complete for track {} of {}'.format(idToTest,max(tracksToTest)), end='\r')
                
            radius_gyration_list.append(radius_gyration_value) 
            asymmetry_list.append(asymmetry_value) 
            skewness_list.append(skewness_value)
            kurtosis_list.append(kurtosis_value) 
            
            trackIntensity_mean.append(np.mean(tracksDF[tracksDF['track_number']==idToTest]['intensity']))
            trackIntensity_std.append(np.std(tracksDF[tracksDF['track_number']==idToTest]['intensity']))    
            
                
        tracksDF['radius_gyration'] = radius_gyration_list
        tracksDF['asymmetry'] = asymmetry_list
        tracksDF['skewness'] = skewness_list
        tracksDF['kurtosis'] = kurtosis_list 
        tracksDF['track_intensity_mean'] = trackIntensity_mean
        tracksDF['track_intensity_std'] = trackIntensity_std
        
        return tracksDF
    
    def getFeaturesForAllTracksinDF(self, tracksDF):
        tracksToTest = tracksDF['track_number'].tolist()
        idTested = []
        fracDim_list = [] 
        netDispl_list = []
        straight_list = []
    
        
        for i in range(len(tracksToTest)):
            idToTest = tracksToTest[i]
            if idToTest not in idTested:
                #get single track
                trackDF = tracksDF[tracksDF['track_number']==idToTest]            
                # Drop any skipped frames and convert trackDF to XY array
                points_array = np.array(trackDF[['x', 'y']].dropna())              
                
                #fractal_dimension calc
                fractal_dimension_value = self.FractalDimension(points_array)
                #net_Dispacement calc
                net_displacement_value, _ = self.NetDisplacementEfficiency(points_array)
                #straightness calc
                _, _, cos_mean_val, _ = self.SummedSinesCosines(points_array)
                
                #update ID tested
                idTested.append(idToTest)
                #print(radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value)
                print('\r' + 'features analysis complete for track {} of {}'.format(idToTest,max(tracksToTest)), end='\r')
            
            #add feature values to lists
            fracDim_list.append(fractal_dimension_value) 
            netDispl_list.append(net_displacement_value)
            straight_list.append(cos_mean_val)
                    
        #update tracksDF        
        tracksDF['fracDimension'] = fracDim_list
        tracksDF['netDispl'] = netDispl_list
        tracksDF['Straight'] = straight_list
        
        return tracksDF
    
    def addLagDisplacementToDF(self, tracksDF):
        #align x and y locations of link
        tracksDF = tracksDF.assign(x2=tracksDF.x.shift(-1))  
        tracksDF = tracksDF.assign(y2=tracksDF.y.shift(-1))
    
        #calculate lag Distance
        tracksDF['x2-x1_sqr'] = np.square(tracksDF['x2']-tracksDF['x'])
        tracksDF['y2-y1_sqr'] = np.square(tracksDF['y2']-tracksDF['y'])      
        tracksDF['distance'] = np.sqrt((tracksDF['x2-x1_sqr']+tracksDF['y2-y1_sqr']))
        
        #mask final track position lags
        tracksDF['mask'] = True
        tracksDF.loc[tracksDF.groupby('track_number').tail(1).index, 'mask'] = False  #mask final location
        
        #get lags for all track locations (not next track)    
        tracksDF['lag'] = tracksDF['distance'].where(tracksDF['mask'])
    
        #add track mean lag distance to all rows
        tracksDF['meanLag'] = tracksDF.groupby('track_number')['lag'].transform('mean')   
        
        #add track length for each track row
        tracksDF['track_length'] = tracksDF.groupby('track_number')['lag'].transform('sum')   
    
        #add 'radius_gyration' (scaled by mean lag displacement)
        tracksDF['radius_gyration_scaled'] = tracksDF['radius_gyration']/tracksDF['meanLag']
    
        #add 'radius_gyration' (scaled by n_segments)
        tracksDF['radius_gyration_scaled_nSegments'] = tracksDF['radius_gyration']/tracksDF['n_segments']
          
        #add 'radius_gyration' (scaled by track_length)
        tracksDF['radius_gyration_scaled_trackLength'] = tracksDF['radius_gyration']/tracksDF['track_length']
            
        print('\r' + 'lags added', end='\r')
         
        return tracksDF
    
    
    def calcFeaturesforFiles(self, tracksDF, minNumberSegments=1):
    
            #add number of segments for each Track row
            tracksDF['n_segments'] = tracksDF.groupby('track_number')['track_number'].transform('count')
                     
            
            if minNumberSegments !=0:
            #filter by number of track segments
                tracksDF = tracksDF[tracksDF['n_segments'] > minNumberSegments]
    
            #add Rg values to df
            tracksDF = self.getRadiusGyrationForAllTracksinDF(tracksDF)
            
            #add features to df
            tracksDF = self.getFeaturesForAllTracksinDF(tracksDF)
            
            #add lags to df
            tracksDF = self.addLagDisplacementToDF(tracksDF)
            
            #add nearest neigbours to df
            #tracksDF = self.getNN(tracksDF)
    
            #### DROP any Unnamed columns #####
            tracksDF = tracksDF[['track_number', 'frame', 'id', 'x','y', 'intensity', 'n_segments', 'track_length','radius_gyration', 'asymmetry', 'skewness',
                                 'kurtosis', 'radius_gyration_scaled','radius_gyration_scaled_nSegments','radius_gyration_scaled_trackLength', 'track_intensity_mean', 'track_intensity_std', 'lag', 'meanLag',
                                 'fracDimension', 'netDispl', 'Straight', 'nnDist', 'nnIndex_inFrame', 'nnDist_inFrame']]
        
        
            #round values
            tracksDF = tracksDF.round({'track_length': 3,
                                       'radius_gyration': 3,
                                       'asymmetry': 3,                                       
                                       'skewness': 3,                                      
                                       'kurtosis': 3,
                                       'radius_gyration_scaled': 3,
                                       'radius_gyration_scaled_nSegments': 3,
                                       'radius_gyration_scaled_trackLength': 3,
                                       'track_intensity_mean': 2,
                                       'track_intensity_std': 2,
                                       'lag': 3,
                                       'meanLag': 3,
                                       'fracDimension': 3,
                                       'netDispl': 3,
                                       'Straight': 3,
                                       'nnDist': 3                                                                              
                                       })
        
        
            return tracksDF
    
    
    def predict_SPT_class(self, df, exptName, level):
        """Computes predicted class for SPT data where
            1:Mobile, 2:Intermediate, 3:Trapped
    
        Args:
            train_data_path (str): complete path to training features data file in .csv format, ex. 'C:/data/tdTomato_37Degree_CytoD_training_feats.csv'
                                   should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('tdTomato_37Degree'),
                                   a 'TrackID' column with unique numerical IDs for each track, and an 'Elected_Label' column derived from human voting.
            pred_data_path (str): complete path to features that need predictions in .csv format, ex. 'C:/data/newconditions/gsmtx4_feature_data.csv'
                                   should be a .csv file with features columns, an 'Experiment' column identifying the unique experiment ('GsMTx4'),
                                   and a 'TrackID' column with unique numerical IDs for each track.
        
        Output:
            .csv file of dataframe of prediction_file features with added SVMPredictedClass column output to pred_data_path parent folder
        """
        def prepare_box_cox_data(data):
            data = data.copy()
            for col in data.columns:
                minVal = data[col].min()
                if minVal <= 0:
                    data[col] += (np.abs(minVal) + 1e-15)
            return data
        
        train_feats = pd.read_csv(Path(train_data_path), sep=',')
        train_feats = train_feats.loc[train_feats['Experiment'] == 'tdTomato_37Degree']
        train_feats = train_feats[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension', 'Elected_Label']]
        train_feats = train_feats.replace({"Elected_Label":  {"mobile":1,"confined":2, "trapped":3}})
        X = train_feats.iloc[:, 2:-1]
        y = train_feats.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        X_train_, X_test_ = prepare_box_cox_data(X_train), prepare_box_cox_data(X_test)
        X_train_, X_test_ = pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_train_), columns=X.columns), pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(X_test_), columns=X.columns)
        
        for col_name in X_train.columns:
            X_train.eval(f'{col_name} = @X_train_.{col_name}')
            X_test.eval(f'{col_name} = @X_test_.{col_name}')
        
        pipeline_steps = [("pca", decomposition.PCA()), ("scaler", StandardScaler()), ("SVC", SVC(kernel="rbf"))]
        check_params = {
            "pca__n_components" : [3],
            "SVC__C" : [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
            "SVC__gamma" : [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1., 5., 10., 50.0],
        }
        
        pipeline = Pipeline(pipeline_steps)
        create_search_grid = GridSearchCV(pipeline, param_grid=check_params, cv=10)
        create_search_grid.fit(X_train, y_train)
        pipeline.set_params(**create_search_grid.best_params_)
        pipeline.fit(X_train, y_train)

        X = df      

        X = X.rename(columns={"radius_gyration" : "radiusGyration",
                          "track_number" : "TrackID",
                          "netDispl" : "NetDispl",
                          "asymmetry" : "Asymmetry",   
                          "kurtosis" : "Kurtosis"                     
                          }) ###GD EDIT
        
        X['Experiment'] = exptName    ###GD EDIT
        
        X = X[['Experiment', 'TrackID', 'NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension']]

        X_feats = X.iloc[0:1, 2:]
        
        X_pred = pipeline.predict(X_feats)
        print(X_pred[0].astype('int'))
        #add classes to RG file ####GD EDIT
        df['SVM'] = X_pred[0].astype('int')
        
        return df


    def testFrameOverlap(self, df, IDlist):
        framesToKeep = []
        newDF = pd.DataFrame()
        #group by frame and sort by longest
        sortedTrackIDs = df.groupby('track_number', as_index=False).count().sort_values('frame',ascending=False, ignore_index=True)['track_number']

        for trackID in sortedTrackIDs:
            trackDF = df[df['track_number'] == trackID]            
            trackDF_frames = trackDF['frame'].tolist()

            check = any(item in trackDF_frames for item in framesToKeep)            
            
            if check == False:
                framesToKeep.extend(trackDF_frames)
                newDF = newDF.append(trackDF)
            
        return newDF    

    def addDiffusiontoDF(self, df):
        
            newDF = pd.DataFrame()
            
            trackList = df['track_number'].unique().tolist()
    
            #iterate through individual tracks
            for track in tqdm(trackList):
                trackDF = df[df['track_number']==track]
                #set positions relative to origin of 0,0
                minFrame = trackDF['frame'].min()
                origin_X = float(trackDF[trackDF['frame'] == minFrame]['x'])
                origin_Y = float(trackDF[trackDF['frame'] == minFrame]['y'])
                trackDF['zeroed_X'] = trackDF['x'] - origin_X 
                trackDF['zeroed_Y'] = trackDF['y'] - origin_Y  
                #generate lag numbers
                trackDF['lagNumber'] = trackDF['frame'] - minFrame
                #calc distance from origin
                trackDF['distanceFromOrigin'] = np.sqrt(  (np.square(trackDF['zeroed_X']) + np.square(trackDF['zeroed_Y']))   )
                    
                #add track results to df
                newDF = pd.concat([newDF, trackDF])
            
            #get squared values            
            newDF['d_squared'] = np.square(newDF['distanceFromOrigin'])
            newDF['lag_squared'] = np.square(newDF['lag'])        
    
            return newDF        

    def addVelocitytoDF(self, df):
            newDF = pd.DataFrame()
            
            trackList = df['track_number'].unique().tolist()
    
            #iterate through individual tracks
            for track in tqdm(trackList):
                trackDF = df[df['track_number']==track]
    
                #add differantial for distance
                diff = np.diff(trackDF['distanceFromOrigin'].to_numpy()) / np.diff(trackDF['lagNumber'].to_numpy())
                diff = np.insert(diff,0,0)
                trackDF['dy-dt: distance'] = diff
        
                #add track results to df
                newDF = pd.concat([newDF, trackDF])
    
        
            #add delta-t for each lag
            newDF['dt'] = np.insert(newDF['frame'].to_numpy()[1:],-1,0) - newDF['frame'].to_numpy()        
            newDF['dt'] = newDF['dt'].mask(newDF['dt'] <= 0, None)
            #instantaneous velocity
            newDF['velocity'] = newDF['lag']/newDF['dt']
            #direction relative to 0,0 origin : 360 degreeas
            degrees = np.arctan2(newDF['zeroed_Y'].to_numpy(), newDF['zeroed_X'].to_numpy())/np.pi*180
            degrees[degrees < 0] = 360+degrees[degrees < 0]        
            newDF['direction_Relative_To_Origin'] =  degrees
            #add mean track velocity 
            newDF['meanVelocity'] = newDF.groupby('track_number')['velocity'].transform('mean')
            
            return newDF
    
    def join(self, df, IDlist):
        
        for trackIDs in IDlist:
            #get new track number
            newTrackID = np.max(df['track_number']) +1
            #filter by IDs
            tracksDF = df[df['track_number'].isin(trackIDs)]
            #filter for non-overlapping tracks
            joinedTrack = self.testFrameOverlap(tracksDF,trackIDs)
            #get old IDs of joined tracks
            oldIDs = joinedTrack['track_number'].unique()
            #delete joinedtracks from original df
            df = df[~df['track_number'].isin(oldIDs)]
            #set newTrack ID
            joinedTrack['track_number'] = newTrackID 
            #drop old columns
            joinedTrack = joinedTrack[['track_number', 'frame', 'id', 'x', 'y', 'intensity', 'Experiment', 'nnDist', 'nnIndex_inFrame', 'nnDist_inFrame']]            
            #run analysis 
            #cal RG - filter for track lengths > minLinkSements 
            rgDF = self.calcFeaturesforFiles(joinedTrack, minNumberSegments=self.minLinkSegments) 
            #resetIndex
            rgDF = rgDF.reset_index()
              
# =============================================================================
#             #classifyTracks(tracksList, trainpath, level='_cutoff_{}'.format(distance)) #uncomment if running multiple cut off values
#             '''This is slow''' #TODO!
#             svmDF = self.predict_SPT_class(rgDF, str(df['Experiment'][0]), level='')
# =============================================================================

            #give an SVM of 0 to indicate joinedtrack
            svmDF = rgDF
            svmDF['SVM'] = 0

            #add diffusion
            diffDF = self.addDiffusiontoDF(svmDF)
            #add velocity
            velDF =  self.addVelocitytoDF(diffDF)

            #append joined track to df
            newDF = df.append(velDF)
                   
        return newDF
    
joinTracks = JoinTracks()

if __name__ == '__main__':
    
    filename = '/Users/george/Data/testing/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity.csv'
    df = pd.read_csv(filename)
    IDlist = [[216, 380, 4612, 6761]]
    
    newDF = joinTracks.join(df, IDlist)
    
    savename = '/Users/george/Data/testing/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_joined.csv'
    newDF.to_csv(savename)
   