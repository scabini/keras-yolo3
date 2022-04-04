#! /usr/bin/env python

import os
import sys
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _main_(args):
    config_path  = args.conf
    
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    input_path = config['test']['test_image_folder']

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 448, 448 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
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

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path[-4:] == '.mp4': # do detection on a video  
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
     
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
