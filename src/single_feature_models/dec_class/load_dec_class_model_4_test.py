import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

"""dec_class"""

"""load the model"""
dec_class_model = xgb.Booster({'nthread':4}) # init model
dec_class_model.load_model('dec_class_model') # load model data

"""load the data: TODO!!!!!!!!!!"""
# should return a data set called data!!
# please take code for load the data in dec_class.py as a reference to load the new test data!!
data = None  # need to be deleted later!!
label = None # need to be deleted later!!

"""use the model to predict the new data"""
dtest = xgb.DMatrix(data.values)
label_pred = dec_class_model.predict(dtest)
acc = accuracy_score(label.values,label_pred.round())
prec = precision_score(label.values, label_pred.round())
re = recall_score(label.values, label_pred.round())
#cm = confusion_matrix(y_test_stra, y_pred)
print(f"Accuracy:\n {100* acc:.2f}%")
print(f"Precision:\n {100* prec:.2f}%")
print(f"Recall:\n {100* re:.2f}%")
#print("confusion matrix:")
#print(cm)
