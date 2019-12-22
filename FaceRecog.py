Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_first')

import pickle
import cv2
import os.path
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utility import *
from webcam_utility import *
np.set_printoptions(threshold=np.nan)




def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[1])) )
    neg_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[2])) )
    basic_loss = pos_dist - neg_dist + alpha
    
    loss = tf.maximum(basic_loss, 0.0)
   
    return loss



def load_FRmodel():
    FRmodel = load_model('models/model.h5', custom_objects={'triplet_loss': triplet_loss})
    return FRmodel

def ini_user_database():
    
    if os.path.exists('database/user_dict.pickle'):
        with open('database/user_dict.pickle', 'rb') as handle:
            user_db = pickle.load(handle)
    else:
        user_db = {}

    return user_db



def add_user_img_path(user_db, FRmodel, name, img_path):
    if name not in user_db:
        user_db[name] = img_to_encoding(img_path, FRmodel)
        with open('database/user_dict.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('User ' + name + ' added successfully')
    else:
        print('The name is already registered! Try a different name.........')
    

def add_user_webcam(user_db, FRmodel, name):
    face_found = detect_face(user_db, FRmodel)

    if face_found:
        resize_img("saved_image/1.jpg")
        if name not in user_db:
            add_user_img_path(user_db, FRmodel, name, "saved_image/1.jpg")
        else:
            print('The name is already registered!')
    else:
        print('no face found in the visible frame.')



def delete_user(user_db, name):
    popped = user_db.pop(name, None)

    if popped is not None:
        print(+ name + ' deleted successfully')
        with open('database/user_dict.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif popped == None:
        print('No such user !!')




def find_face(image_path, database, model, threshold=0.6):
    
    encoding = img_to_encoding(image_path, model)

    min_dist = 99999
   
    for name in database:
        
        dist = np.linalg.norm(np.subtract(database[name], encoding))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        print("User not in database.")
        identity = 'Unknown'
    else:
        print(+ str(identity) + ", L2 distance: " + str(min_dist))

    return min_dist, identity


def do_face_recognition(user_db, FRmodel, threshold=0.7, save_loc="saved_image/1.jpg"):
    face_found = detect_face(user_db, FRmodel)

    if face_found:
        resize_img("saved_image/1.jpg")
        find_face("saved_image/1.jpg", user_db, FRmodel, threshold)
    else:
        print('There was no face found')



def main():
    FRmodel = load_FRmodel()
    print('\n\nModel loaded...')

    user_db = ini_user_database()
    print('User database loaded')
    
    ch = 'y'
    while(ch == 'y' or ch == 'Y'):
        user_input = input(
            '\nEnter choice \n1. Realtime Face Recognition\n2. Recognize face\n3. Add or Delete user\n4. Quit\n')

        if user_input == '1':
            os.system('cls' if os.name == 'nt' else 'clear')
            detect_face_realtime(user_db, FRmodel, threshold=0.6)

        elif user_input == '2':
            os.system('cls' if os.name == 'nt' else 'clear')
                        do_face_recognition(user_db, FRmodel, threshold=0.6,
                                save_loc="saved_image/1.jpg")

        elif user_input == '3':
            os.system('cls' if os.name == 'nt' else 'clear')
            add_ch = input()
            name = input('Enter the name of the person')

            if add_ch == '1':
                img_path = input(
                'Enter the image name with extension')
                add_user_img_path(user_db, FRmodel, name, 'images/' + img_path)
            elif add_ch == '2':
                add_user_webcam(user_db, FRmodel, name)
            elif add_ch == '3':
                delete_user(user_db, name)
            else:
                print('Invalid choice')

        elif user_input == '4':
            return

        else:
            print('Invalid choice\n')

        ch = input('Continue ? yes or no')
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == main():
    main()
