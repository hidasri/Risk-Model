# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 11:46:21 2015

@author: SXK9800
"""


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from time import time
#==============================================================================
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import VarianceThreshold
#==============================================================================

def get_dummies_custom(dataset):
    list_objects=[col for col in dataset if dataset.dtypes[col]=='object']
    for lo in list_objects:
        dummy_df=pd.get_dummies(dataset[lo].astype(str))
        print lo,list(dummy_df.columns.values)
        dummy_df.columns=[lo+'_'+str(xx) for xx in list(dummy_df.columns.values)]
        del dataset[lo]
        dataset=pd.concat([dataset,dummy_df],axis=1)
    return dataset

root_dir="S:/lou01/CODA/Behavioral Health Analytics/Personal Folders/Sri/Suicide_model/"
id_variable_name='pers_gen_key'
target_variable_name='SUICIDE_FLAG'

train=pd.read_csv(root_dir+"SIC_10_90_TRAIN_IMP_VAR.csv");
test=pd.read_csv(root_dir+"SIC_10_90_TEST_IMP_VAR.csv");

length_train= len(train.index)
train_test=pd.concat([train,test])
train_test_dummies=get_dummies_custom(train_test)

train_dummies=train_test_dummies[:length_train]
test_dummies=train_test_dummies[length_train:]   

id_train=train_dummies[id_variable_name]
Y_train=train_dummies[target_variable_name]
del train_dummies[id_variable_name]
del train_dummies[target_variable_name]

id_test=test_dummies[id_variable_name]
Y_test=test_dummies[target_variable_name]
del test_dummies[id_variable_name]
del test_dummies[target_variable_name]


#Model 1: Support Vector Machines
pipeline=Pipeline([('clf_svm',SGDClassifier(loss='hinge', penalty='l2')])
parameters = dict(clf_svm__alpha=[1,.1, .01,.001,.0001,.00001,.000001,.0000001,.00000001])
#Model 2: Logistic regression
pipeline=Pipeline([('clf_lr',SGDClassifier(loss='loss', penalty='l2')])
parameters = dict(clf_lr__alpha=[1,.1, .01,.001,.0001,.00001,.000001,.0000001,.00000001])
#Model 3: Random Forest Classifier
pipeline=Pipeline([('clf_rfc',RandomForestClassifier(n_estimators=3)])
parameters = dict(clf_rfc__criterion=['gini'])


grid_search = GridSearchCV(pipeline, param_grid=parameters,cv=10,scoring='f1')




print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(train_dummies, Y_train)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    
#Batch mode
#    To add:
#        LDA, QDA, Neural Network, decision trees, naive bayes

pipeline_list=[Pipeline([('clf_svm',SGDClassifier(loss='hinge', penalty='l2')]),#SVM
               Pipeline([('clf_lr',SGDClassifier(loss='loss', penalty='l2')]),#Logistic Regression
               Pipeline([('clf_rfc',RandomForestClassifier(n_estimators=100)]),#Random forest classifier
               Pipeline([('clf_gbc',GradientBoostingClassifier(n_estimators=100)])#Gradient boosting classifier
              ]
parameters_list=[dict(clf_svm__alpha=[1,.1, .01,.001,.0001,.00001,.000001,.0000001,.00000001]),#SVM
                 dict(clf_lr__alpha=[1,.1, .01,.001,.0001,.00001,.000001,.0000001,.00000001]),#Logistic Regression
                 dict(clf_rfc__criterion=['gini','entropy']),#Random forest classifier
                 dict(clf_gbc__loss=['deviance','exponential'])#Gradient boosting classifier
                ]
print("Performing grid search...")
for i in range(len(parameters_list)):
    pipeline=pipeline_list[i]
    parameters=parameters_list[i]
    grid_search = GridSearchCV(pipeline, param_grid=parameters,cv=10,scoring='f1')
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(train_dummies, Y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

    
    
    
