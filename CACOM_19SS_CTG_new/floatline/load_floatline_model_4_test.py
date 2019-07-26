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

def check_data_info(data):
	# check the overall info of the data, can only be applied to pandas dataframe
	return data.describe()

def down_sample(data,start=-1,end=-7200,step=-4):
	#param start: index for the start point
	#param end: index for the end point
	#param step: sample interval
	return data[start:end:step]

def replace_0_with_nan(data):
	data[data==0] = np.nan
	return data

def interpolate_data(data):
	# Note: the interpolation can only be used to interpolate the NaN data
	new_data = data.interpolate()
	return new_data

def drop_nan(data):
	# Note: this step can be used to drop the last few nan values,
	# which original are the last few 0 values
	new_data = data.dropna()
	return new_data

def fourier_transform(data):
	new_data = np.fft.fft(data)
	return new_data

def plot_data(data):
	plt.plot(data)
	plt.show()


"""floatline"""

"""load the model"""
floatline_model = xgb.Booster({'nthread':4}) # init model
floatline_model.load_model('floatline_model') # load model data

"""load the data: TODO!!!!!!!!!!"""
# should return a data set called data!!
# please take load_all_data function in floatline.py as a reference to load the new test data!!
data = None  # need to be deleted later!!
label = None # need to be deleted later!!

"""use the model to predict the new data"""
dtest = xgb.DMatrix(data.values)
label_pred = floatline_model.predict(dtest)
acc = accuracy_score(label.values,label_pred.round())
prec = precision_score(label.values, label_pred.round())
re = recall_score(label.values, label_pred.round())
#cm = confusion_matrix(y_test_stra, y_pred)
print(f"Accuracy:\n {100* acc:.2f}%")
print(f"Precision:\n {100* prec:.2f}%")
print(f"Recall:\n {100* re:.2f}%")
#print("confusion matrix:")
#print(cm)