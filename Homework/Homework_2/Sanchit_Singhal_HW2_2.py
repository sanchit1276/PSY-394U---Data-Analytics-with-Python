## Author : Sanchit Singhal
## Date : 19/02/2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA,PCA

### 1. Load the data
ghgData = pd.read_fwf('GHG_Data.txt',header=None)

#fix configuration of data to have time on x-axis for time-series analysis
ghgData = ghgData.transpose()

### 2. Determine the number of components by performing a PCA
pca = PCA()
ghgPC = pca.fit_transform(ghgData)
nIC = np.min(np.where(np.cumsum(pca.explained_variance_ratio_)>0.9))+1
#it was determined that 9 components explained atleast 90% of variabiity

### 3. Perform ICA
ica = FastICA(n_components=nIC)
ghgIC = ica.fit_transform(ghgData)

### 4. Plot the time courses of the independent components
plt.figure(figsize=[8,4])
for iIC in range(nIC):
    plt.plot(ghgIC[:,iIC], label = 'IC ' + str(iIC+1))
plt.title('Time Courses of the Independent Components')
plt.xlabel('Time')
plt.ylabel('Variation in Greenhouse Gas Concentration')
plt.legend()
plt.show()
