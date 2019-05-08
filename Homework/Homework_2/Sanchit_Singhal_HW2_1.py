## Author : Sanchit Singhal
## Date : 19/02/2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

### 1. Load the data
LeafData = pd.read_csv('LeafData.csv',header=None)

# Extracting and Normalizing Features
LeafFeatures = np.array(LeafData.iloc[:,1:])
LeafFeaturesNorm = StandardScaler().fit_transform(LeafFeatures)

# Extract and Converting Target values for comparison with clusters
mymap = {'Acer Capillipes':0, 'Betula Austrosinensis':1, 'Castanea Sativa':2, 'Cytisus Battandieri':3,
    'Eucalyptus Glaucescens':4,'Ginkgo Biloba':5,'Ilex Cornuta':6,'Magnolia Salicifolia':7,
        'Populus Adenopoda':8,'Tilia Platyphyllos':9}
y = LeafData.applymap(lambda s: mymap.get(s) if s in mymap else s)
y = np.array(y.iloc[:,0])

### 2. Dimension Reduction
LeafPCA = PCA(n_components=64)
LeafPCs = LeafPCA.fit_transform(LeafFeaturesNorm)

# Scree Plot
plt.plot(np.arange(1,65), LeafPCA.explained_variance_ratio_)
plt.title('Scree Plot')
plt.xlabel('Component number')
plt.ylabel('Proportion variance explained')
plt.show()

# Using Scree Plot, it was determined that there is an elblow around 20 PCs
LeafPC = LeafPCs[:,:21]

### 3. Perform Clustering
km = KMeans(n_clusters=10)  # defining the clustering object
km.fit(LeafPC)  # actually fitting the data
y_clus = km.labels_   # clustering info resulting from K-means

#Below section is for plotting, If not wanted - please comment out
plt.figure(figsize=[12,4])
#Plot Clusters from K-means
plt.subplot(121)
plt.scatter(LeafPC[:,0],LeafPC[:,1],c=y_clus,marker='+')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters from K-means')
#Plot True Clusters
plt.subplot(122)
plt.scatter(LeafPC[:,0],LeafPC[:,1],c=y,marker='+')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('True Clusters')
plt.show()

### 4. Evaluate the clustering performance
print('ARI=',adjusted_rand_score(y, y_clus),sep='')
print('AMI=',adjusted_mutual_info_score(y, y_clus),sep='')
