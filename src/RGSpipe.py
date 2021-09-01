#############
###Imports###
#############
import numpy as np
from glob import glob
import pandas as pd
import keras
import pickle
import tensorflow as tf
import socket
import time
import cv2
from functions import split_3_coordinates
from functions import mediapipe_inference
from functions import callibration_picture

#Socket definition
ip_receiver = '127.0.0.1'
port_receiver = 5005
experiment = 'Experiment_07'

#Path of scaler and nn weights:
model = keras.models.load_model('../models/'+experiment+'/model.h5')
scaler = pickle.load(open('../models/'+experiment+'/scaler.pkl','rb'))


##########################
###Callibration picture###
##########################

frame = callibration_picture()
detected_keypoints, frame = mediapipe_inference(frame)
#If there is one NaN in callibration picture (joint not detected properly) or no keypoints have been detected --> Another picture is taken
while pd.DataFrame(detected_keypoints).T.isna().sum().sum()>= 1 or len(detected_keypoints) == 0: 
    print('Joints have not been detected properly - Another picture will be taken \n')
    frame = callibration_picture()
    detected_keypoints,frame = mediapipe_inference(frame)

df = pd.DataFrame(detected_keypoints).T

#############
###RGSpipe###
#############

#Frame number - 0 is the callibration picture, the following the detected webcam frames
n = 0

#Initialize stream capture
cap = cv2.VideoCapture(0)

prueba_time_mp = []

try: 
    while cap.isOpened():
        n = n+1
        sucess,frame = cap.read()
        frame = cv2.resize(frame,(1920,1080))

        #MediaPipe inference
        start1 = time.time()
        detected_keypoints, frame = mediapipe_inference(frame)
        
        
        end1 = time.time()
        prueba_time_mp.append(end1-start1)
        #print(prueba_time_mp)

        df = df.append(pd.DataFrame(detected_keypoints).T)
        df = df[[2,3,6,7,10,11,0,1,4,5,8,9]]
        df = df.fillna(method='ffill')

        if all(isinstance(n,float) for n in list(df.iloc[n])) == True:

        #NN inference
    
            X = np.array([list(df.iloc[n])]).astype(float)
            X_scaled = scaler.transform(X)
            z_predicted = model.predict(X_scaled)
            df_pred_3d = pd.DataFrame(split_3_coordinates(z_predicted))
            
            #Sending udp output
            msg = bytes(str(df_pred_3d.iloc[0].tolist()),'utf-8')
            print(f'Sending {msg} to {ip_receiver}:{port_receiver}')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(msg,(ip_receiver, port_receiver))
            print(df_pred_3d)
        
except KeyboardInterrupt:
    print('Camera is closed')
    cap.release()