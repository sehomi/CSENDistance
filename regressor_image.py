import os
import numpy as np
import matplotlib as plt
import argparse
import scipy.io
import cv2 as cv
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

from csen_regressor import model

def formScaler():

    x_train = data['x_train'].astype('float32')
    x_dic = data['x_dic'].astype('float32')

    m =  x_train.shape[1]
    n =  x_train.shape[2]

    x_dic = np.reshape(x_dic, [len(x_dic), m * n])
    x_train = np.reshape(x_train, [len(x_train), m * n])
    
    scaler = StandardScaler().fit(np.concatenate((x_dic, x_train), axis = 0))

    return scaler


def preprocess(imgDir, bbox, scaler):
    function_name = 'tf.keras.applications.' + feature_type
    model = eval(function_name + "(include_top=False, weights='imagenet', input_shape=(64, 64, 3), pooling='max')")

    im = cv.imread(imgDir)

    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[4])
    y2 = int(bbox[5])

    Object = cv.resize(im[y1:y2, x1:x2, :], (64, 64))
    Object = np.expand_dims(cv.cvtColor(Object, cv.COLOR_BGR2RGB), axis=0)

    function_name = 'tf.keras.applications.' + feature_type[:8].lower() + '.preprocess_input'
    Object = eval(function_name + '(Object)')
 
    f = model.predict(Object)

    f = np.transpose(f).astype(np.double)

    phi = data['phi'].astype('float32')
    Proj_M = data['Proj_M'].astype('float32')

    Y2 = np.matmul(phi, f)
    Y2 = Y2 / np.linalg.norm(Y2)

    prox_Y2 = np.matmul(Proj_M, Y2)

    prox_Y2 = scaler.transform(prox_Y2.T).T

    x_test = np.reshape(prox_Y2, (1, 15, 80))

    x_test = np.expand_dims(x_test, axis=-1)

    return x_test

modelType = 'CSEN'
feature_type = 'VGG19'
weights = True

MR = '0.5'

outName  = 'results/CSEN/' + feature_type + '_mr_' + MR

weightsDir = 'weights/' + modelType + '/'

modelFold = model.model()
    
weightPath = weightsDir + feature_type + '_' + MR + '_' + str(1) + '.h5'

# Testing and performance evaluations.
modelFold.load_weights(weightPath)

data = scipy.io.loadmat('CSENdata-2D/VGG19_mr_0.5_run1.mat')

scaler = formScaler()

# Path for the data and annotations.
kittiData = 'kitti-data/'
# Note that KITTI provides annotations only for the training since it is a challenge dataset.
imagePath = kittiData + 'training/image_2/'
df = pd.read_csv(kittiData + 'annotations.csv')

err = []
counter = 0

for idx, row in df.iterrows():
    if row['class'] != 'Pedestrian':
        continue

    counter += 1
    if counter>40:
        break

    imageName = kittiData + 'training/image_2/' + row['filename'].replace('txt', 'png')
    im = cv.imread(imageName) # Load the image.
    
    # Object Location.
    x1 = int(row['xmin'])
    y1 = int(row['ymin'])
    x2 = int(row['xmax'])
    y2 = int(row['ymax'])
    fts = preprocess(imageName,[x1,y1,x2,y1,x2,y2,x1,y2], scaler)
    y_pred = modelFold.model.predict(fts)

    cv.rectangle(im, (x1,y1), (x2,y2), (255,0,0), 2)

    dist = float(row['zloc'])
    cv.putText(im, 'p: {:.1f} r:{:.1f}'.format(y_pred[0][0],dist), (x1,y1-10), cv.FONT_HERSHEY_SIMPLEX,\
                            1, (255,0,0), 2, cv.LINE_AA)
    # print(y_pred)
    cv.imwrite('images/{}'.format(row['filename'].replace('txt', 'png')), im)

    err.append(np.linalg.norm(y_pred[0][0]-dist))

err = np.array(err)
print('mean error: {:.2f}'.format(err.mean()))

del modelFold