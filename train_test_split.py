# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:29:29 2022

I used this script to split the data into validation foolders

@author: scabini
"""

import os
import json
import shutil
import numpy as np
from sklearn.model_selection import KFold


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

dataset = 'C:/Users/Svartox/Documents/datasets/GUVs_PP-SC/GUVs_all/'
images = getListOfFiles(dataset + 'images/')
annot  = getListOfFiles(dataset + 'annotations/')

folds=10
kf = KFold(n_splits=folds, shuffle=True)


i=0
for train_index, test_index in kf.split(images):  
    i=i+1
    fold = dataset + 'fold' +str(i)
    os.makedirs(fold, exist_ok=True)
    os.makedirs(fold + '/train/', exist_ok=True)
    os.makedirs(fold + '/test/', exist_ok=True)  
    
    os.makedirs(fold + '/train/images/', exist_ok=True)
    os.makedirs(fold + '/train/annotations/', exist_ok=True)
    
    os.makedirs(fold + '/test/images/', exist_ok=True)
    os.makedirs(fold + '/test/annotations/', exist_ok=True)
    
    for j in range(len(images)):
        img_name = images[j].split('/')[-1]
        anot_name = annot[j].split('/')[-1]
        if j in train_index:
            shutil.copyfile(images[j], fold + '/train/images/' + img_name)
            shutil.copyfile(annot[j], fold + '/train/annotations/' + anot_name)
        else:
            shutil.copyfile(images[j], fold + '/test/images/' + img_name)
            shutil.copyfile(annot[j], fold + '/test/annotations/' + anot_name)
            
    fold_info = {
    "domain" : "tutswiki",
    "language" : "python",
    "date" : "11/09/2020",
    "topic" : "config file"
    }
    myJSON = json.dumps(fold_info)
    with open("tutswiki.json", "w") as jsonfile:
        jsonfile.write(myJSON)
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            