Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 26 2018, 23:26:24) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> import cv2
import os, sys
import pickle
import numpy as np
import numpy
from os import path
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from PIL import Image
from PIL import ImageEnhance
import scipy.misc

parent_path = os.path.dirname(os.getcwd())
parent_path = os.path.dirname(parent_path)
data_path = parent_path + "/Images/Ariba_file"
model_path = parent_path + "/common/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(model_path)


datagen = ImageDataGenerator(
                 featurewise_center = False,             
                 samplewise_center  = False,             
                 featurewise_std_normalization = False,  
                 samplewise_std_normalization  = False,                    
                 rotation_range = 20,                    
                 width_shift_range  = 0.2,
		 height_shift_range = 0.2,               
                 horizontal_flip = True,                
                 vertical_flip = False)                  

def detectFace(detector,image_path, image_name):
    imgAbsPath = image_path + image_name
    img = cv2.imread(imgAbsPath)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    w = img.shape[1]
    faces = detector.detectMultiScale(gray, 1.1,5,0,(w//2,w//2))

    resized_im = 0

    if len(faces) == 1:
        face = faces[0]
        croped_im = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2],:]
        resized_im = cv2.resize(croped_im, (224,224))

        if resized_im.shape[0] != 224 or resized_im.shape[1] != 224:
            print("invalid shape")

        
    else:
        print(image_name+" error " + str(len(faces)))
    return resized_im


def randomUpdate(img):

    img = scipy.misc.toimage(img)

    
    rotate = random.random() * 30 - 30
    image_rotated = img.rotate(rotate)

    
    enh_bri = ImageEnhance.Brightness(image_rotated)
    bright = random.random() * 0.8 + 0.6
    image_brightened = enh_bri.enhance(bright)



    enh_col = ImageEnhance.Color(image_contrasted)
    color = random.random() * 0.6 + 0.7
    image_colored = enh_col.enhance(color)

    enhance_im = np.asarray(image_colored)

    return enhance_im

lable_distribution = []

rating_files = ['images1.csv',
                'images2.csv',
                'images3.csv',
                'images4.csv',
                'images5.csv']

pre_vote_image_name = ''
pre_vote_image_score1_cnt = 0
pre_vote_image_score2_cnt = 0
pre_vote_image_score3_cnt = 0
pre_vote_image_score4_cnt = 0
pre_vote_image_score5_cnt = 0

for rating_file_name in rating_files:

    rating_file = open(rating_path+rating_file_name, 'r')

    lines = rating_file.readlines();
    lines.pop(0)
    lineIdx = 0

    for line in lines:

        line = line.strip().split(',')
        lineIdx += 1;
        curr_row_image_name = line[1]
        score = int(line[2])

        if pre_vote_image_name == '':
            pre_vote_image_name = curr_row_image_name
        if (curr_row_image_name != pre_vote_image_name) or (lineIdx == lines.__len__()):
            total_vote_cnt = pre_vote_image_score1_cnt + pre_vote_image_score2_cnt + pre_vote_image_score3_cnt + pre_vote_image_score4_cnt + pre_vote_image_score5_cnt
            score1_ld = pre_vote_image_score1_cnt
            score2_ld = pre_vote_image_score2_cnt 
            score3_ld = pre_vote_image_score3_cnt 
            score4_ld = pre_vote_image_score4_cnt 
            score5_ld = pre_vote_image_score5_cnt 
                    im = detectFace(face_cascade, data_path, pre_vote_image_name)

            if isinstance(im, numpy.ndarray):
                normed_im = (im - 127.5) / 127.5

                ld = []
                ld.append(score1_ld)
                ld.append(score2_ld)
                ld.append(score3_ld)
                ld.append(score4_ld)
                ld.append(score5_ld)
                lable_distribution.append([pre_vote_image_name, normed_im, ld]))

            pre_vote_image_name = curr_row_image_name
            pre_vote_image_score1_cnt = 0
            pre_vote_image_score2_cnt = 0
            pre_vote_image_score3_cnt = 0
            pre_vote_image_score4_cnt = 0
            pre_vote_image_score5_cnt = 0
        if score == 1:
            pre_vote_image_score1_cnt += 1
        elif score == 2:
            pre_vote_image_score2_cnt += 1
        elif score == 3:
            pre_vote_image_score3_cnt += 1
        elif score == 4:
            pre_vote_image_score4_cnt += 1
        elif score ==5:
            pre_vote_image_score5_cnt += 1

    rating_file.close()


data_split_index = int(lable_distribution.__len__() - lable_distribution.__len__()*0.1)

random.shuffle(lable_distribution)
test_lable_distribution = lable_distribution[data_split_index:]
train_lable_distribution = lable_distribution[:data_split_index]


train_data_len = train_lable_distribution.__len__()
for i in range(0, train_data_len):
   
    im = train_lable_distribution[i][1]
    enhance_im = randomUpdate(im)
    enhance_normed_im = (enhance_im - 127.5) / 127.5

    train_lable_distribution.append([pre_vote_image_name, enhance_normed_im, ld])

   
