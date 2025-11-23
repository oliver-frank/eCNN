# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 23:12:34 2021

@author: olive
"""

# =============================================================================
# Libraries importieren 
# =============================================================================

import cv2
import numpy as np
import tensorflow as tf # Version 2.4.0


# =============================================================================
# Model konvertieren und Classifier laden
# =============================================================================

# Konvertiere das bereits tranierten, ursprüngliche Modell entsprechend der
# tf-lite-Konvertierung 
# converter = tf.lite.TFLiteConverter.from_saved_model('model.model')
# tflite_model = converter.convert()
# Speichere das konvertierte Modell
# open("converted_model.tflite", "wb").write(tflite_model)


# Hier noch einmal definieren für Trennung des Codes
img_size = 150
# Modell laden
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()
# Classifier, der direkt nach Gesichtern in gegebenen Bild sucht 
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 


# =============================================================================
# Video starten - Model und Classifier anwenden 
# =============================================================================

# Starte Videoaufnahme  
# 0 = WebCam extern; 1 = Webcam intern 
video_capture = cv2.VideoCapture(0) 
# Dictonaries für einfachere Schreibweise 
labels_dict = {0:'NO MASK',1:'MASK'}
color_dict  = { 0:(0,0,255),1:(0,255,0)}

while(True):
    # repeat Frame Aufnahme bis sie unterbrochen wird 
    ret,frame = video_capture.read() 
    # Grayscaling, um Rechenleistung zu verringern (von 3 (RGB) auf 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    # Klassifizierte Bereiche in Variable schließen
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        # Grayscaled image 
        face_img = gray[y:y+w,x:x+h]
        # Größe des durch Classifier ausgesuchten Bildes auf img_size mappen
        resized = cv2.resize(face_img,(img_size,img_size))
        # Pixelwert-Normalisierung
        normalized = resized/255.0
        # Matrix zu np-array shapen und 1er Vektor anhängen
        reshaped = np.reshape(normalized,(1,img_size,img_size,1)) 
        # Details des interpreter modells
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # converted die eingegebenen Daten in die richtige Form für das Modell
        input_data = np.array(reshaped, dtype=np.float32)
        # gib dem Modell die Daten 
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Lass das Modell die Daten verarbeiten 
        interpreter.invoke()
        # catche die verarbeitenden Daten 
        result = interpreter.get_tensor(output_details[0]['index'])
        # gibt den größten Wert eines arrays entlang der definierten Achse aus 
        # 0: y-Achse 1: x-Achse 
        label = np.argmax(result,axis=1)[0]
        # Schließt Bildbereich des Classifiers in Rechteck ein, das je nach 
        # Klassifizierung des faces durch das Modell grün (mit Maske: 1) oder 
        # rot (ohne Maske: 0) angezeigt wird
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)    
        cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, \
            0.8,(255,255,255),2)
    # WebCam Bild wird in Fenster auf Screen gezeigt   
    cv2.imshow('Video',frame) 
    # (1) Gibt fließendes Video aus
    # (0) würde nur vereinzelte Frames herausgeben 
    key=cv2.waitKey(1) 
    # Break, wenn esc 
    if(key==27):   
        break;
# Alle Fenster und Videos schließen
cv2.destroyAllWindows()     

video_capture.release()     
