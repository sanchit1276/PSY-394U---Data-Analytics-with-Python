import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# loadin the data and standardizing it
seedData = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
seedFeatures = np.array(seedData.iloc[:,:7])
seedTargets = np.array(seedData.iloc[:,7]) - 1 # starting from zero
targetNames = ['Kama','Rosa','Canadian']
targetColors = ['red','blue','green']
seedFeaturesNorm = StandardScaler().fit_transform(seedFeatures)


# Performing the linear discriminant analysis

# Plotting the first and second LDs
