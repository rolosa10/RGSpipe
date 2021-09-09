import mediapipe as mp
import numpy as np
import cv2
import time
import variables

#Initialize the Mediapipe module with its corresponding parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0,min_detection_confidence=0.4,min_tracking_confidence=0.5)

def split_3_coordinates(values_array): 
    output = []
    for i in range(0, len(values_array)):
        m = 0
        r = 3
        frame_coordinates = []
        for j in range(0,int(len(values_array[i])/3)):
            frame_coordinates.append(values_array[i][m:r])
            m = r
            r = r+3
        output.append(frame_coordinates)
    return(output)

def mediapipe_inference(frame):
    
    results = pose.process(frame)
    frame_list1 = []
    point_coords_landmark_interest = []
    start_index = 0
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            #!!!!! If landmark visibility is not 0.4 save a NaN and this value will be replaced by the previous recorded value !!!!
            #Reference & first value is the callibraition picture 
            if start_index in variables.indices_landmark_interest:
                if landmark.visibility>0.4:
                    image_hight, image_width, _ = frame.shape
                    frame = cv2.circle(frame, (int(landmark.x*image_width), int(landmark.y*image_hight)), radius=5, color = (0,255,0), thickness=-1)   
                    frame_list1.append(landmark.x*image_width)
                    frame_list1.append(landmark.y*image_hight)
                    point_coords_landmark_interest.append((int(round(landmark.x*image_width)),int(round(landmark.y*image_hight))))
                    start_index = start_index+1
                else:
                    frame_list1.append(np.NaN)
                    frame_list1.append(np.NaN)
                    start_index = start_index+1
            else: 
                start_index = start_index+1
   
    else:
        frame_list1 = np.zeros(variables.number_keypoints_to_detect*2)
        frame_list1[:] = np.nan    
        
    return(frame_list1,frame)
        
def callibration_picture():
    print('A picture will be taken in 2 secs. Please show all your upper body')
    #time.sleep(2)
    for i in range(1,2):
        cap1 = cv2.VideoCapture(0)
        success, frame1 = cap1.read()
        frame1 = cv2.resize(frame1,(1920,1080))
        cap1.release()
    return(frame1)

def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img