import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import copy
import pandas as pd
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
	# replace the 0 value with np.nan, so that interpolate and drop function can be applied
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

def plot_and_save(data,name):
	plt.plot(data)
	plt.savefig('/Users/zoe/Desktop/baseline/'+'{}'.format(name)+'.jpg')# change to your own path
	plt.close()

def classify(data, label, save=True):
	num_fold = 0
	num_of_splits = 5
	x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=42)
	fold_data = np.zeros((num_of_splits, x_train.shape[0]))
	folds = KFold(n_splits=num_of_splits, shuffle=True, random_state=42)
	clf = XGBClassifier()
	### training phase
	for train_idx, val_idx in folds.split(x_train,y_train):  #<class 'numpy.ndarray'>
		train_copy = copy.deepcopy(train_idx)
		val_copy = copy.deepcopy(val_idx)
		fold_data[num_fold, :] = np.concatenate((train_copy, val_copy), axis=0)
		x_train_stra, x_val_stra = x_train[train_idx, :], x_train[val_idx, :]
		y_train_stra, y_val_stra = y_train[train_idx], y_train[val_idx]
		print("------training phase------")
		print()
		print("Fold:", num_fold)
		print()
		clf.fit(x_train_stra, y_train_stra)
		y_pred = clf.predict(x_val_stra)
		acc = accuracy_score(y_val_stra, y_pred)
		prec = precision_score(y_val_stra, y_pred)
		re = recall_score(y_val_stra, y_pred)
		cm = confusion_matrix(y_val_stra, y_pred)
		print(f"Accuracy:\n {100* acc:.2f}%")
		print(f"Precision:\n {100* prec:.2f}%")
		print(f"Recall:\n {100* re:.2f}%")
		print("confusion matrix:")
		print(cm)
		num_fold = num_fold + 1
	### test phase
	print()
	print("------test phase------")
	test_pred = clf.predict(x_test)
	acc_t = accuracy_score(y_test, test_pred)
	prec_t = precision_score(y_test, test_pred)
	re_t = recall_score(y_test, test_pred)
	cm_t = confusion_matrix(y_test, test_pred)
	print(f"Accuracy:\n {100* acc_t:.2f}%")
	print(f"Precision:\n {100* prec_t:.2f}%")
	print(f"Recall:\n {100* re_t:.2f}%")
	print("confusion matrix:")
	print(cm_t)
	if save:
		return clf,fold_data


def load_all_data(path):
	# load handpicked feature data from path
	# pre-fixed parameter
	num_point = 1000 # only 1000 valid data points from the last second of the whole data set are utilized, this parameter can be changed later
	num_ft = 30 # only the first 30 items of outcomes of the fourier transform are utilized, this parameter can be changed later
	case_list = os.listdir(path)
	case_list_copy = copy.deepcopy(case_list)
	random.shuffle(case_list_copy)
	case_list_copy = case_list_copy[0:95]  ######## Due to the limitation of acidic samples, to balance the input data, only load 95 samples each, can be changed later
	data_baseline = np.zeros((len(case_list),num_point))
	ft_baseline = np.zeros((len(case_list),num_ft*2))
	num_baseline_case = 0
	for i_case, case_name in enumerate(case_list_copy):
		raw_data = xr.open_dataset(path + '/' + case_name).to_array(dim='feature').transpose().to_pandas()
		# one row := one sample
		raw_data_baseline = raw_data['baseline']
		new_data_baseline = drop_nan(interpolate_data(replace_0_with_nan(down_sample(raw_data_baseline)))).iloc[0:num_point].values
		if len(new_data_baseline) == 1000:
			data_baseline[i_case,:] = new_data_baseline
			#plot_and_save(new_data_baseline, case_name)
			ft_data = fourier_transform(new_data_baseline).reshape((1,-1))[:,:num_ft]
			ft_baseline[i_case,:] = np.column_stack((ft_data.real, ft_data.imag))
			num_baseline_case += 1
	return data_baseline[:num_baseline_case,:], ft_baseline[:num_baseline_case,:]

"""load the data"""
abnormal_path = '/Users/zoe/Desktop/CACOM/20190121 DECFIT analysed test data_JG/data/train/Fallgruppe_60'
ab_baseline, ab_baseline_ft = load_all_data(abnormal_path)
normal_path = '/Users/zoe/Desktop/CACOM/20190121 DECFIT analysed test data_JG/data/train/Kontrollgruppe_60'
nor_baseline, nor_baseline_ft  = load_all_data(normal_path)

"""if you want to check the data information of one specific case, use the check_data_info function, e.g.:"""
# case0 = pd.DataFrame(data = ab_baseline[0])
# check_data_info(case0)
"""if you want to save the np array data to a csv file, use the following code to save it, e.g.:"""
# np.savetxt("ab_baseline_ft.csv", ab_baseline_ft, delimiter=",")
"""if you want to see the plot of the data, use the plot_data function, e.g.:"""
# np.plot(data)

all_data_baseline = np.concatenate((ab_baseline_ft, nor_baseline_ft[:ab_baseline_ft.shape[0],:]), axis=0)
all_label_baseline = np.concatenate((np.ones((len(ab_baseline_ft),1)), np.zeros((len(ab_baseline_ft),1))), axis=0).ravel()
#np.savetxt("shuffled_baseline_data.csv", shuffled_baseline_data, delimiter=",")
model,fold_data = classify(all_data_baseline, all_label_baseline)

"""save the model"""
model.save_model('baseline_model')
