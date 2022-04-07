# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:59:00 2022

@author: scabini
"""
import os

# folds = ['fold'+str(i) for i in range(1,5)]
# folds = ['fold'+str(i) for i in range(5,8)]
folds = ['fold'+str(i) for i in range(8,11)]

dataset = '/home/scabini/Experiments/datasets/Vesiculas_GUVs/GUVs_all/splited/'
output = '/home/scabini/Experiments/results/Doutorado/Vesiculas_GUVs/predictions/'

for fold in folds:
    print("training at", fold, "...")
    file = dataset +  fold + '/' + fold
    os.system("python train.py -c " + file + ".json")
    
    print("predicting/writing imgs at", fold, "...")    
    os.system("python predict.py -c " + file + ".json -o " + output)



