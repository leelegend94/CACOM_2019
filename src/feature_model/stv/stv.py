import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, accuracy_score


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

def classify(data, label, save=True):
	x_train, x_test, y_train, y_test = train_test_split(data, label)
	clf = XGBClassifier()
	clf.fit(x_train,y_train)
	y_pred = clf.predict(x_test)
	acc = accuracy_score(y_test,y_pred)
	prec = precision_score(y_test, y_pred)
	print(f"Accuracy:\n {100* acc:.2f}%")
	print(f"Precision:\n {100* prec:.2f}%")
	if save:
		return clf

def load_all_data(path):
	# load handpicked feature data from path
	# pre-fixed parameter
	num_point = 1000 # only 1000 valid data points from the last second of the whole data set are utilized, this parameter can be changed later
	num_ft = 30 # only the first 30 items of outcomes of the fourier transform are utilized, this parameter can be changed later
	case_list = os.listdir(path)
	data_stv = np.zeros((len(case_list),num_point))
	ft_stv = np.zeros((len(case_list),num_ft*2))
	num_stv_case = 0
	for i_case, case_name in enumerate(case_list):
		raw_data = xr.open_dataset(path + '/' + case_name).to_array(dim='feature').transpose().to_pandas()
		# one row := one sample
		raw_data_stv = raw_data['stv']
		new_data_stv = drop_nan(interpolate_data(replace_0_with_nan(down_sample(raw_data_stv)))).iloc[0:num_point].values
		if len(new_data_stv) == 1000:
			data_stv[i_case,:] = new_data_stv
			ft_data = fourier_transform(new_data_stv).reshape((1,-1))[:,:num_ft]
			ft_stv[i_case,:] = np.column_stack((ft_data.real, ft_data.imag))
			num_stv_case += 1
	return data_stv[:num_stv_case,:], ft_stv[:num_stv_case,:]

"""load the data"""
abnormal_path = '/Users/zoe/Desktop/CACOM/20190121 DECFIT analysed test data_JG/data/train/Fallgruppe_60'
ab_stv, ab_stv_ft = load_all_data(abnormal_path)
normal_path = '/Users/zoe/Desktop/CACOM/20190121 DECFIT analysed test data_JG/data/train/Kontrollgruppe_60'
nor_stv, nor_stv_ft  = load_all_data(normal_path)

"""if you want to check the data information of one specific case, use the check_data_info function, e.g.:"""
# case0 = pd.DataFrame(data = ab_stv[0])
# check_data_info(case0)
"""if you want to save the np array data to a csv file, use the following code to save it, e.g.:"""
# np.savetxt("ab_stv.csv", ab_fhr, delimiter=",")
"""if you want to see the plot of the data, use the plot_data function, e.g.:"""
# plot_data(ab_stv)

"""train the model"""
all_data_stv = np.concatenate((ab_stv_ft, nor_stv_ft), axis=0)
all_label_stv = np.concatenate((np.zeros((len(ab_stv_ft),1)), np.ones((len(nor_stv_ft),1))), axis=0)
# shuffle the index
stv_index = [i for i in range(len(ab_stv_ft)+len(nor_stv_ft))]
random.shuffle(stv_index)
shuffled_stv_data = all_data_stv[stv_index]
shuffled_stv_label = all_label_stv[stv_index].ravel()
#np.savetxt("shuffled_stv_data.csv", shuffled_stv_data, delimiter=",")
#np.savetxt("shuffled_stv_label.csv", shuffled_stv_label, delimiter=",")
model = classify(shuffled_stv_data, shuffled_stv_label)
# for reference: Accuracy:88.74%, Precision:89.95%

"""save the model"""
model.save_model('stv_model')