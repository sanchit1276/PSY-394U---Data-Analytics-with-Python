## Author : Sanchit Singhal
## Date : 03/25/2019

## import required libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

### Load the data

# read file
with open('university_data.txt', 'r') as f:
    raw_data = f.read().lower()

# split observations
split_data = raw_data.split("(def-")

# remove duplicates and nulls
filter_data = []
for data in split_data:
    if "duplicate" in data:
        pass
    elif len(data) == 0:
        pass
    else:
        filter_data.append(data)

# handle multiple academic-emphasis features
raw_feature_data = []
for data in filter_data:
    temp=[]
    temp = data.split("(academic-emphasis")
    raw_feature_data.append(temp[0])

# split features and assign to feature key
feature_data = []
for data in raw_feature_data:
    temp = data.split("(")
    feature_data.append(temp)
features = ["instance", "state ","control","location","no-of-students thous:","male:female ratio:",
            "student:faculty ratio:","sat verbal", "sat math","expenses thous$:","percent-financial-aid",
            "no-applicants thous:","percent-admittance","percent-enrolled","academics scale:","social scale:",
            "quality-of-life scale:"]
feature_split_data = []
for x in feature_data:
    temp = {}
    for y in x:
        for feature in features:
            if feature in y:
                value = y.split(feature)
                temp[feature] =value[1]
    feature_split_data.append(temp)

# preprocess data to obtain usable format for models
preprocessed_data =[]
for x in feature_split_data:
    temp ={}
    for key,value in x.items():
        value = value.replace(")","");
        value = value.replace(" ","");
        value = value.replace("%","")
        value = value.replace("private:roman-catholic","private")
        value = value.replace("city","state")
        value = value.replace("\n","")
        value = value.replace("act-21","0")
        value = value.replace("act-15","0")
        value = value.replace("n/a","0")
        value = value.replace("?","0")
        value = value.replace("na","0")
        if "+" in value:
            value = int(value.replace("+",""))
        elif "1-5" in value:
            value1 = value.replace("1-5","")
            value = int(value1)
        elif "-" in value and len(value)<7:
            data1 = value.split("-")
            if data1[1] == '':
                data1[1] = "0"
            value = (int(data1[0]) + int(data1[1]))/2
        elif ":" in value and len(value)<9:
            data2 = value.split(":")
            if data2[1] != "0":
                value = int(data2[0])/int(data2[1])
            else:
                value = int(data2[0])
        try:
            value =float(value)
        except ValueError:
            value = value
        temp[key] = value
    preprocessed_data.append(temp)

# convert standarized data to dataframe
data_for_dataframe=[]
for x in preprocessed_data:
    temp = x
    for feature in features:
        if feature not in x:
            temp[feature] = None
    data_for_dataframe.append(temp)

df =pd.DataFrame(data_for_dataframe)

# impute NAs with mean
df.fillna(df.mean(), inplace=True)
df = df.replace(to_replace='None', value=np.nan).dropna()

# change target values from state to public (since we are categorizing public vs private)
df['control'] = df['control'].replace(to_replace='state', value='public')

## Encoding categorical variables

# the target

controlLE = LabelEncoder()
y = controlLE.fit_transform(df['control'])
y_class = controlLE.classes_

# the features

locationLE = LabelEncoder()
locationx = locationLE.fit_transform(df['location'])
locationx_class = locationLE.classes_

academicLE = LabelEncoder()
academicx = academicLE.fit_transform(df['academics scale:'])
academicx_class = academicLE.classes_

socialLE = LabelEncoder()
socialx = socialLE.fit_transform(df['social scale:'])
socialx_class = socialLE.classes_

qualityLE = LabelEncoder()
qualityx = qualityLE.fit_transform(df['quality-of-life scale:'])
qualityx_class = qualityLE.classes_

# recreate feature variables (not using university name, state, academic emphasis)
X = np.vstack([locationx,academicx,socialx,qualityx,df['male:female ratio:'],df['no-of-students thous:']
               ,df['sat verbal'],df['sat math'],df['expenses thous$:'],df['percent-financial-aid']
               ,df['no-applicants thous:'],df['percent-admittance'],df['percent-enrolled']
               ,df['student:faculty ratio:']]).T

### Feature Selection and Classification Model:Random Forest w/variable 'Expenses'

# build base model pipline
rf = Pipeline([
    ('normalization',StandardScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
    ('classification',RandomForestClassifier())
])

# list hyperparameters of model
param_rf = {'classification__criterion': ['entropy','gini'], 'classification__n_estimators': [1,5,7,10],
            'classification__max_depth':[4,5,6,7,8,9,10,11,12,13], 'classification__min_samples_leaf': [1,2,3,4,5,6,7,8]}

# using a grid search to build hyper parameters to test
grid_rf = GridSearchCV(rf, param_rf, cv=5)
grid_rf.fit(X,y)

# Optimal set of hyperparameters and its corresponding accuracy score
print ("Optimal Parameters of RF with Variable 'expense': %s" % grid_rf.best_params_)
print ("Model Accuracy of RF with Variable 'expense': %s \n" % grid_rf.best_score_)

### Feature Selection and Classification Model:Random Forest w/o variable 'Expenses'

# recreate feature variables without the variable 'expense' (not using university name, state, academic emphasis)
X = np.vstack([locationx,academicx,socialx,qualityx,df['male:female ratio:'],df['no-of-students thous:']
               ,df['sat verbal'],df['sat math'],df['percent-financial-aid']
               ,df['no-applicants thous:'],df['percent-admittance'],df['percent-enrolled']
               ,df['student:faculty ratio:']]).T

# build base model pipline
rf = Pipeline([
    ('normalization',StandardScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
    ('classification',RandomForestClassifier())
])

# list hyperparameters of model
param_rf = {'classification__criterion': ['entropy','gini'], 'classification__n_estimators': [1,5,7,10],
            'classification__max_depth':[4,5,6,7,8,9,10,11,12,13], 'classification__min_samples_leaf': [1,2,3,4,5,6,7,8]}

# using a grid search to build hyper parameters to test
grid_rf = GridSearchCV(rf, param_rf, cv=5)
grid_rf.fit(X,y)

# Optimal set of hyperparameters and its corresponding accuracy score
print ("Optimal Parameters of RF without Variable 'expense': %s" % grid_rf.best_params_)
print ("Model Accuracy of RF without Variable 'expense': %s" % grid_rf.best_score_)
