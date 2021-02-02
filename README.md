# CACOM 2019
Fetal health prediction based on CTG data using XGBoost. Help doctors making a decision on cesarean section(C-Section).
---
## Introduction
We designed a novel algorithm to predict fetal condition before delivery. The algorithm utilizes Fetal Heart Rate (FHR) as the input, which is a long time-series data. Instead of using RNN or LSTM empirically, we do the Fourier transformation to the time series data as pre-processing. The frequency-domain spectrum is used as the feature for the classifier.

## Prerequirements
The following packages are required for the project:
* sklearn, scipy
* xarray
* pandas
* matplotlib
* xgboost

## Run

Models based on different features are located in the folder src/single_feature_models.

Optimized model with normalized Fourier transform, Automatic hyper parameter tuning is located in the folder src/fhr_optimized_model

Choose the desired folder and run the python3 script.

## Result

The trained XGBoost model is tested on a testset. The performance is shown below:
* Accuracy: 69.77%
* Precision: 76.47%
* Recall: 59.09%