import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import copy
import pandas as pd 
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier
from xgboost import to_graphviz
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks

def check_data_info(data):
    # check the overall info of the data, can only be applied to pandas dataframe
    return data.describe()

def down_sample(data,start=-1,end=-7200,step=-8):
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
    new_data = np.fft.fft(data)/len(data)
    return new_data

def plot_data(data):
    plt.plot(data)
    plt.show()

def plot_and_save(data,name):
    plt.plot(data)
    plt.savefig('./CACOM_data_plot/fhr_out_ft/'+'{}'.format(name)+'.jpg')# change to your own path
    plt.close()

def test(clf, x_test, y_test):
    print("------test phase------")
    print("Sample Number:", len(x_test))
    print()
    test_pred = clf.predict(x_test)
    acc_t = accuracy_score(y_test, test_pred)
    prec_t = precision_score(y_test, test_pred)
    re_t = recall_score(y_test, test_pred)
    cm_t = confusion_matrix(y_test, test_pred)
    print("Accuracy:\n {:.2f}%".format(100* acc_t))
    print("Precision:\n {:.2f}%".format(100* prec_t))
    print("Recall:\n {:.2f}%".format(100* re_t))
    print("confusion matrix:")
    print(cm_t)



def load_all_data2(path):
    # load handpicked feature data from path
    # pre-fixed parameter
    num_point = 7200 # only 1000 valid data points from the last second of the whole data set are utilized, this parameter can be changed later
    #num_ft = 30 # only the first 30 items of outcomes of the fourier transform are utilized, this parameter can be changed later
    num_peaks = 10
    num_peaks_i = 3
    case_list = os.listdir(path)
    case_list = list(filter(lambda x: x.endswith('nc'), case_list))

    random.shuffle(case_list)
    case_list_copy = case_list[0:96]  ######## Due to the limitation of acidic samples, to balance the input data, only load 95 samples each, can be changed later
    #data_fhr = np.zeros((len(case_list_copy),num_point))
    #ft_fhr = np.zeros((len(case_list_copy),num_ft*2)) # since we need both real and imag part, so the col num should be doubled
    ft_fhr = []
    data_fhr = []
    feature = []
    num_fhr_case = 0
    for i_case, case_name in enumerate(case_list_copy):
        raw_data = xr.open_dataset(path + '/' + case_name).to_array(dim='feature').transpose().to_pandas()

        raw_data_fhr = raw_data['corrected'].tolist()[-num_point:]  #there is no need to interpolate the data, since the corrected data is availiable, downsample is not necessary as well, because we will apply fourier transform

        new_data_fhr = raw_data_fhr
        if len(new_data_fhr) == num_point:
            #data_fhr.append(new_data_fhr)
            #plot_and_save(new_data_fhr, case_name)
            ft_data = fourier_transform(new_data_fhr)
            tmp_real = smooth(ft_data.real,51)
            tmp_imag = smooth(ft_data.imag,51)
            d_real = down_sample(tmp_real)[:50]
            d_imag = down_sample(tmp_imag)[:50]
            peaks_real, _ = find_peaks(abs(tmp_real),distance=10,prominence=0.005)
            peaks_imag, _ = find_peaks(abs(tmp_imag),distance=10,prominence=0.005)
            if(len(peaks_real) >= num_peaks or len(peaks_imag) >= num_peaks_i):
                ampl_real = abs(tmp_real[peaks_real])
                ampl_imag = abs(tmp_real[peaks_imag])
                real_idx = np.argsort(ampl_real)[:num_peaks]
                imag_idx = np.argsort(ampl_imag)[:num_peaks_i]
                feature.append(np.array(list(ampl_real[real_idx])+list(peaks_real[real_idx])+list(ampl_imag[imag_idx])+list(peaks_imag[imag_idx])))
                #ft_fhr.append(np.column_stack((down_sample(tmp_real[:int(num_point/2)]).reshape(1,-1), down_sample(tmp_imag[:int(num_point/2)]).reshape(1,-1))))
                ft_fhr.append(np.column_stack((d_real.reshape(1,-1), d_imag.reshape(1,-1))))
                data_fhr.append(new_data_fhr)
                #plot_and_save(ft_fhr[i_case,:], case_name)
                num_fhr_case += 1

    feature = np.stack(feature,axis=0)
    ft_fhr = np.stack(np.squeeze(ft_fhr,axis=0))
    #for x in feature:
    #    print(x.shape)
    #return data_fhr, ft_fhr
    return ft_fhr, feature

def load_all_data(path):
    # load handpicked feature data from path
    # pre-fixed parameter
    num_point = 1000 # only 1000 valid data points from the last second of the whole data set are utilized, this parameter can be changed later
    num_ft = 30 # only the first 30 items of outcomes of the fourier transform are utilized, this parameter can be changed later
    case_list = os.listdir(path)
    case_list = list(filter(lambda x: x.endswith('.nc'), case_list))
    case_list_copy = case_list
    random.shuffle(case_list_copy)
    case_list_copy = case_list_copy[0:95]  ######## Due to the limitation of acidic samples, to balance the input data, only load 95 samples each, can be changed later
    data_fhr = np.zeros((len(case_list),num_point))
    ft_fhr = np.zeros((len(case_list),num_ft*2)) # since we need both real and imag part, so the col num should be doubled
    num_fhr_case = 0
    for i_case, case_name in enumerate(case_list_copy):
        raw_data = xr.open_dataset(path + '/' + case_name).to_array(dim='feature').transpose().to_pandas()
        # one row := one sample
        raw_data_fhr = raw_data['fhr-out']
        # after drop_nan function, data type: <class 'pandas.core.series.Series'>,
        # so we need to process the data using .iloc[].values, now the data type should be numpy array
        new_data_fhr = drop_nan(interpolate_data(replace_0_with_nan(down_sample(raw_data_fhr)))).iloc[-num_point:].values
        if len(new_data_fhr) == 1000:
            data_fhr[i_case,:] = new_data_fhr
            #plot_and_save(new_data_fhr, case_name)
            ft_data = fourier_transform(new_data_fhr).reshape((1,-1))[:,:num_ft]
            ft_fhr[i_case,:] = np.column_stack((ft_data.real.reshape(1,-1), ft_data.imag.reshape(1,-1)))
            #plot_and_save(ft_fhr[i_case,:], case_name)
            num_fhr_case += 1
    return data_fhr[:num_fhr_case,:], ft_fhr[:num_fhr_case,:]

def smooth(data, window_size=5):
    return np.convolve(data,np.ones(window_size),'valid')/window_size

def optmize(train_X, train_Y):
    '''
    parameters = {
              'max_depth': [10, 15, 20],
              'learning_rate': [0.05, 0.1, 0.15],
              'n_estimators': [500, 1000, 2000],
              'min_child_weight': [0, 2, 5, 10, 20],
              'max_delta_step': [0, 0.2, 0.6, 1, 2],
              'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
              'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
              'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
              'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    }
    '''
    parameters = {
            #'max_depth': [10, 20],
            #'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            #'max_delta_step': [0.2, 0.4]
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'scale_pos_weight': [0.6, 0.8, 1]
            }
    
    xlf = XGBClassifier(max_depth=10,
            learning_rate=0.05,
            n_estimators=300,
            silent=True,
            objective='binary:logistic',
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0.2,
            subsample=0.85,
            colsample_bytree=0.7,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=1440,
            missing=None,
            tree_method='gpu_hist')
        
    gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=10)
    gsearch.fit(train_X, train_Y)
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return gsearch.best_estimator_

print("loading and pre-processing")
abnormal_path = '/home/zhenyu/workspace/cacom/CACOM_2019/DataSet/Fallgruppe_60'
ab_fhr, ab_fhr_ft = load_all_data2(abnormal_path)
normal_path = '/home/zhenyu/workspace/cacom/CACOM_2019/DataSet/Kontrollgruppe_60'
nor_fhr, nor_fhr_ft  = load_all_data2(normal_path)
print(ab_fhr.shape)
print(nor_fhr.shape)
#all_data_fhr = np.concatenate((ab_fhr_ft, nor_fhr_ft), axis=0)
all_data_fhr = np.concatenate((ab_fhr, nor_fhr), axis=0)
all_label_fhr = np.concatenate((np.zeros((len(ab_fhr_ft),1)), np.ones((len(nor_fhr_ft),1))), axis=0).ravel()
print(len(nor_fhr_ft))
print(len(ab_fhr_ft))
x_train, x_test, y_train, y_test = train_test_split(all_data_fhr, all_label_fhr)
print("optimizing...")
model = optmize(x_train,y_train)
print("training...")
model.fit(x_train, y_train)
model.save_model('model.xgb')
print("model saved!")
test(model,x_test,y_test)
s = to_graphviz(model)
tree_idx = 0
s.render('tree_plot_'+str(tree_idx)+'.gv',view=True,num_trees=tree_idx)