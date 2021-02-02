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


"""load the data"""
path = '/Users/zoe/Desktop/CACOM/total_labeled_dec.csv' # change the path to your own path
dec_class_data = pd.read_csv(path)
ab_dec_data =  dec_class_data[dec_class_data['acidose']==1] # type(ab_dec_data):pandas frame ;len(ab_dec_data): 86
nor_dec_data = dec_class_data[dec_class_data['acidose']==0]

"""train the model"""
all_data_dec = np.concatenate((ab_dec_data.values[:,:-1], nor_dec_data.values[:len(ab_dec_data),:-1]), axis=0)
# note that label for abnormal case is 0, label for normal case is 1
all_label_dec = np.concatenate((np.ones((len(ab_dec_data),1)), np.zeros((len(ab_dec_data),1))), axis=0).ravel()
#np.savetxt("shuffled_dec_class_data.csv", shuffled_dec_data, delimiter=",")
#np.savetxt("shuffled_dec_class_label.csv", shuffled_dec_label, delimiter=",")
model,fold_data = classify(all_data_dec, all_label_dec)

"""save the model"""
model.save_model('dec_class_model') # save the trained model, name 'dec_model' can be changed