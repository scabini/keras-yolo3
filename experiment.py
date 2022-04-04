# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:59:00 2022

@author: scabini
"""
import os

folds = ['fold'+str(i) for i in range(1,11)]

for fold in folds:
    print("training for fold ", fold, "...")
    os.system("python train.py -c " + fold + "_no_outliers.json")


for fold in folds:
    print("predicting/writing imgs for fold ", fold, "...")
    os.system("python predict.py -c " + fold + "_no_outliers.json -o output_noOutliers/")



os.system("python train.py -c fold999_allData.json")

