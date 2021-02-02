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

"""load the data"""
path = '/Users/zoe/Desktop/CACOM/total_labeled_dec.csv' # change the path to your own path
dec_class_data = pd.read_csv(path)
ab_dec_data =  dec_class_data[dec_class_data['acidose']==1] # type(ab_dec_data):pandas frame ;len(ab_dec_data): 86
nor_dec_data = dec_class_data[dec_class_data['acidose']==0]

"""train the model"""
all_data_dec = np.concatenate((ab_dec_data.values[:,:-1], nor_dec_data.values[:,:-1]), axis=0)
# note that label for abnormal case is 0, label for normal case is 1
all_label_dec = np.concatenate((np.zeros((len(ab_dec_data),1)), np.ones((len(nor_dec_data),1))), axis=0)
dec_index = [i for i in range(len(ab_dec_data)+len(nor_dec_data))] # shuffle the index
random.shuffle(dec_index)
shuffled_dec_data = all_data_dec[dec_index]
shuffled_dec_label = all_label_dec[dec_index].ravel()
#np.savetxt("shuffled_dec_class_data.csv", shuffled_dec_data, delimiter=",")
#np.savetxt("shuffled_dec_class_label.csv", shuffled_dec_label, delimiter=",")
model = classify(shuffled_dec_data, shuffled_dec_label)
# for reference: Accuracy:88.52%, Precision:89.81%

"""save the model"""
model.save_model('dec_class_model') # save the trained model, name 'dec_model' can be changed