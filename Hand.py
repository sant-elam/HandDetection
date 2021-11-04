# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:00:37 2021

------ HAND TRACKER -------
@author: 
"""
import cv2 as cv
import mediapipe as mp

hand_object = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils

drwlandmarks = draw_utils.DrawingSpec( (255,0, 0), thickness= 2, circle_radius=3)
drwconnections= draw_utils.DrawingSpec( (0, 0, 255), thickness= 2)


STATIC_IMAGE = False
MAX_NO_OF_HANDS = 2
DETECT_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5


hand_model = hand_object.Hands( static_image_mode = STATIC_IMAGE,
                               max_num_hands = MAX_NO_OF_HANDS,
                               min_detection_confidence = DETECT_CONFIDENCE,
                               min_tracking_confidence=TRACKING_CONFIDENCE)

path = 'C:/MEDIA_PIPE/VID-20211028-WA0002.mp4'

capture = cv.VideoCapture(path)

while True:
    
    result, image_org = capture.read()
    
    if result:
        
        image = cv.cvtColor(image_org, cv.COLOR_BGR2RGB)
        
        output = hand_model.process(image)
        
        if output.multi_hand_landmarks:
            
            for hands in output.multi_hand_landmarks:
                draw_utils.draw_landmarks(image_org,
                                          hands,
                                          hand_object.HAND_CONNECTIONS,
                                          landmark_drawing_spec=drwlandmarks,
                                          connection_drawing_spec=drwconnections)
                
        cv.imshow("HAND", image_org)
        if cv.waitKey(30) & 255 == 27:
            break
        
capture.release()
cv.destroyAllWindows()
                
                
                
                
