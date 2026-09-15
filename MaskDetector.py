# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:21:32 2021

@author: olive
"""

import cv2
import numpy as np
from keras.models import load_model



img_size = 150



model = load_model('model.model') #load the best model



faceCascade=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)  #starts the webcam
labels_dict = {0:'NO MASK',1:'MASK'}
color_dict  = { 0:(0,0,255),1:(0,255,0)}



while(True):
    ret,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        face_img = gray[y:y+w,x:x+h]
        resized = cv2.resize(face_img,(img_size,img_size))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,img_size,img_size,1))
        result = model.predict(reshaped)
        label = np.argmax(result,axis=1)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow('Video',frame)
    key=cv2.waitKey(1)
    
    if(key==27):
        break;
        
cv2.destroyAllWindows()
video_capture.release()

