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

# dataset = 'C:/Users/Svartox/Documents/datasets/GUVs_PP-SC/GUVs_all/'
dataset = '/home/scabini/Experiments/datasets/Vesiculas_GUVs/GUVs_all/raw_data/'


images = getListOfFiles(dataset + 'images/')
images.sort()

annot  = getListOfFiles(dataset + 'annotations/')
annot.sort()

output = '/home/scabini/Experiments/datasets/Vesiculas_GUVs/GUVs_all/splited/'

shutil.rmtree(output)

os.makedirs(output, exist_ok=True)

folds=10
kf = KFold(n_splits=folds, shuffle=True)


gpu = {1:'0', 2:'0', 3:'0', 4:'0',
       5:'1', 6:'1', 7:'1',
       8:'2', 9:'2', 10:'2'}

i=0
for train_index, test_index in kf.split(images):  
    i=i+1
    fold = output + 'fold' +str(i)
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
        print(img_name, anot_name)
        if j in train_index:
            shutil.copyfile(images[j], fold + '/train/images/' + img_name)
            shutil.copyfile(annot[j], fold + '/train/annotations/' + anot_name)
        else:
            shutil.copyfile(images[j], fold + '/test/images/' + img_name)
            shutil.copyfile(annot[j], fold + '/test/annotations/' + anot_name)
            
            
    fold_info = {
        "model" : {
    	"name": "YOLOv3_fold" + str(i),
            "min_input_size":       352,
            "max_input_size":       448,
            "anchors":              [11,17, 15,23, 20,29, 25,36, 31,46, 41,56, 60,78, 104,127, 282,272],
            "labels":               ["changed", "control", "outlier"]
        },
    
        "train": {
            "train_image_folder":   fold + '/train/images/',
            "train_annot_folder":   fold + '/train/annotations/',
            "cache_name":           fold + '/fold' + str(i) + '.pkl',
    
            "train_times":          5,
            "batch_size":           1,
            "learning_rate":        1e-3,
            "nb_epochs":            100,
            "warmup_epochs":        3,
            "ignore_thresh":        0.5,
            "gpus":                 gpu[i],
    
            "grid_scales":          [1,1,1],
            "obj_scale":            5,
            "noobj_scale":          1,
            "xywh_scale":           1,
            "class_scale":          1,
    
    	"tensorboard_dir":      "logs",
            "saved_weights_name":   '/home/scabini/Experiments/results/Doutorado/Vesiculas_GUVs/trained_YOLOmodels/fold' + str(i) + '_model.h5',
            "debug":                True
        },
    
        "valid": {
            "valid_image_folder":   "",
            "valid_annot_folder":   "",
            "cache_name":           '/home/scabini/Experiments/results/Doutorado/Vesiculas_GUVs/trained_YOLOmodels/fold' + str(i) + '_valid_cache.pkl',
            "valid_times":          1
        },
        "test": {
            "test_image_folder":   fold + '/test/images/',
            "test_annot_folder":   fold + '/test/annotations/',
            "cache_name":          '/home/scabini/Experiments/results/Doutorado/Vesiculas_GUVs/trained_YOLOmodels/fold' + str(i) + '_test_cache.pkl',
            "valid_times":          1
        }
    }

    myJSON = json.dumps(fold_info, indent=4)
    with open(fold + "/fold" + str(i)+ ".json", "w") as jsonfile:
        jsonfile.write(myJSON)
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            