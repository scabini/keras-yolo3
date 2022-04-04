# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:59:00 2021

@author: Svartox
"""
import os

folds = ['fold'+str(i) for i in range(1,11)]

for fold in folds:
    print("starting ", fold, " WITHOUT outliers ...")
    os.system("python train.py -c " + fold + "_no_outliers.json")

