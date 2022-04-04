# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:49:09 2021

@author: Svartox

"""

import os

folds = ['fold'+str(i) for i in range(1,11)]

for fold in folds:
    # print("starting ", fold, " WITHOUT outliers ...")
    os.system("python predict.py -c " + fold + "_with_outliers.json -o output_withOutliers/")

for fold in folds:
    # print("starting ", fold, " WITHOUT outliers ...")
    os.system("python predict.py -c " + fold + "_no_outliers.json -o output_noOutliers/")
