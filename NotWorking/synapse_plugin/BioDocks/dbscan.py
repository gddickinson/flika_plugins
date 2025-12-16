import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def dbscan(xyData, eps=500, min_samples = 10, plot=False):
    '''
    Function for applying scikit DBSCAN to 3D imaging data

    Parameters
    ----------
    xyData : TYPE
        DESCRIPTION.
    eps : TYPE, optional
        DESCRIPTION. The default is 500.
    min_samples : TYPE, optional
        DESCRIPTION. The default is 10.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    labels : TYPE
        DESCRIPTION.
    n_clusters_ : TYPE
        DESCRIPTION.
    n_noise_ : TYPE
        DESCRIPTION.

    '''
    # #############################################################################
    # Generate sample data
    X = xyData
    
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples,
                metric='euclidean', metric_params=None,
                algorithm='auto', leaf_size=30,
                p=None, n_jobs=-1).fit(X) #eps - max distance between 2 points in cluster (in nanometers) 
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    

    #identify clusters with less than min_samples parameter (DBSCAN can return clusters with fewer points than set by parameter)
    unique, counts = np.unique(labels, return_counts=True)
    labels_count = dict(zip(unique, counts))

    #get labels with counts less than min_samples 
    filterDict = {k:v for (k,v) in labels_count.items() if v < min_samples}
    labelsToFilter = filterDict.keys()
    
    #convert cluster labels with low numbers of points to noise (-1)
    if bool(filterDict): 
        labels = [-1 if x in labelsToFilter else x for x in labels]
    
    #recount clusters
    unique, counts = np.unique(labels, return_counts=True)
    labels_count = dict(zip(unique, counts))    

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    smallestGroup = min(labels_count, key=labels_count.get)  
    largestGroup = max(labels_count, key=labels_count.get) 
   
     
    print('Mininum number of samples parameter: %d' % min_samples)    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)    
    print('Number of points counted in smallest cluster: %d' %labels_count[smallestGroup])    
    print('Number of points counted in largest cluster: %d' %labels_count[largestGroup])  

    
    # #############################################################################
    if plot:    
        # Plot result
        import matplotlib.pyplot as plt
        
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
        
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        
    return labels, n_clusters_, n_noise_