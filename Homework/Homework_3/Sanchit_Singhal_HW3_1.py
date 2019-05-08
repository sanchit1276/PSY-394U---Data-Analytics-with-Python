## Author : Sanchit Singhal
## Date : 03/25/2019

## import required libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

### Load the data

df = pd.read_csv('tae.csv',header=None)
df = df.rename(index=int, columns={0: "english", 1: "instructor", 2: "course", 3:"semester", 4:"size", 5:'eval'})
featureNames = ['english','instructor','course','semester','size']
targetNames = ['Low','Medium','High']

## Encoding categorical variables (all but size)

# the target

evalLE = LabelEncoder()
y = evalLE.fit_transform(df['eval'])
y_class = evalLE.classes_

# the features

englishLE = LabelEncoder()
englishx = englishLE.fit_transform(df['english'])
englishx_class = englishLE.classes_

instructorLE = LabelEncoder()
instructorx = instructorLE.fit_transform(df['instructor'])
instructorx_class = instructorLE.classes_

courseLE = LabelEncoder()
coursex = courseLE.fit_transform(df['course'])
coursex_class = courseLE.classes_

semesterLE = LabelEncoder()
semesterx = semesterLE.fit_transform(df['semester'])
semesterx_class = semesterLE.classes_

# recreate features

X = np.vstack([englishx,instructorx,coursex,semesterx,df['size']]).T

### Classification Model:Random Forest

# build base model pipeline
rf = Pipeline([
    ('normalization',StandardScaler()),
    ('classification',RandomForestClassifier())
])

# list hyperparameters of model
param_rf = {'classification__criterion': ['entropy','gini'], 'classification__n_estimators': [1,5,7,10],
            'classification__max_depth':[4,5,6,7,8,9,10,11,12,13], 'classification__min_samples_leaf': [1,2,3,4,5,6,7,8]}

# using a grid search to build hyper parameters to test
grid_rf = GridSearchCV(rf, param_rf, cv=5)
grid_rf.fit(X,y)

# Optimal set of hyperparameters and its corresponding accuracy score
print ("Optimal Parameters of RF: %s" % grid_rf.best_params_)
print ("Model Accuracy of RF: %s" % grid_rf.best_score_)
