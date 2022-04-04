#! /usr/bin/env python

import argparse
import sys
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['KMP_WARNINGS'] = 'off'
os.environ['KMP_AFFINITY'] = 'none'


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:        
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################  
    valid_ints, labels = parse_voc_annotation(
        config['test']['test_annot_folder'], 
        config['test']['test_image_folder'], 
        config['test']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    # print("Number of arguments: ", len(sys.argv))
    # print("The arguments are: " , str(sys.argv))
    
    os.environ['KMP_WARNINGS'] = 'off'
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    # print("RUNNING ON GPU ", sys.argv[0])
    print("RUNNING ON GPU ", config['train']['gpus'])
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    sys.stderr = stderr

    backend_config = tf.ConfigProto()
    backend_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    backend_config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=backend_config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    infer_model = load_model(config['train']['saved_weights_name'])

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.6f}'.format(average_precision))
    print('mAP: {:.6f}\n'.format(sum(average_precisions.values()) / len(average_precisions)))           

    for label, average_precision in average_precisions.items():
        print('{:.6f}\n'.format(average_precision))
    print('{:.6f}'.format(sum(average_precisions.values()) / len(average_precisions))) 

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')    
    
    args = argparser.parse_args()
    _main_(args)
