# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:29:34 2021

@author: Svartox
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

folds = ['fold'+str(i) for i in range(1,11)]

losses = []

# losses = np.zeros((1,100))
for fold in folds:
    file = 'results/' + fold + "_with_outliers.pickle"
    with open(file, 'rb') as f:
        loss, yolo_layer_1_loss, yolo_layer_2_loss, yolo_layer_3_loss = pickle.load(f)
        loss = np.asarray(loss)
        losses.append(loss)
        # losses[0,0:np.size(loss)] = losses[0,0:np.size(loss)] + loss
        plt.figure(0)
        # plt.rcParams.update({'font.size': 16})
       
        plt.plot(loss)
        
plt.xlabel('epochs')
plt.ylabel('validation accuracy (%)')
        
        # plt.yticks([85, 90, 95, 100], ['85', '90', '95', '100'])
        # plt.ylim([80, 104])
plt.grid()
        # plt.legend()
        # plt.tight_layout()
plt.show()
        # plt.savefig("plots_Experiment3/architecture_comparison.pdf", dpi=500)


# losses = losses/4    
