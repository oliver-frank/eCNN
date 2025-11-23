# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:55:37 2021

@author: OliveTree
"""

# pip install --user google-assistant-sdk[samples]
# pip install opencv-python
# pip install os
# pip install numpy
# pip install tensorflow
# pip install -U scikit-learn
# pip install matplotlib
# pip install h5py


# =============================================================================
# Step 1: Importieren der notwendigen Bibliotheken
# =============================================================================


import cv2,os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils  
from keras.models import Sequential 
from keras.layers import  Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split
from keras.models import load_model
# from matplotlib import pyplot as plt


# =============================================================================
# Step 2: Pfad definieren und Kategorien labeln
# =============================================================================


# Pfad in dem die Trainingsdaten abgespeichert sind 
data_path = r'C:\YourDataPath'
categories = os.listdir(data_path) 
# Zuordnung von 0 und 1 an categories -> 0: without_mask  1: with_mask 
labels = [i for i in range(len(categories))] 
label_dict = dict(zip(categories,labels))

# Kontrolle
print(label_dict)
print(categories)
print(labels)


# =============================================================================
# Step 3: Daten- und Labelliste erstellen 
# =============================================================================


img_size = 150
# leere Listen erstellen
data_list = []      
label_list = []
# Iteriert durch alle categories -> 2 
for category in categories:
    # ertsellt einen Pfad pro Iteration und öffnet so beide Mask-Ordner
    folder_path = os.path.join(data_path,category)  
    # fügt alle Tainingsobjekte aus dem jeweiligen Ordner in eine Liste
    img_names = os.listdir(folder_path) 

    print(folder_path)
    print(len(img_names))
    
    # iteriert durch alle Objekte der beiden Listen von img_names 
    for img_name in img_names: 
        # erstellt einen Pfad pro Iteration und öffnet so jedes Bild in beiden 
        # Ordnern 
        img_path = os.path.join(folder_path,img_name)
        img = cv2.imread(img_path)  

        try:   
            # gray-scaled jedes image und macht so aus 3 RGB-Werten -> 
            # 1 GrayScale-Wert für niedrigere Rechenkapazität 
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            # jedes image wird auf 150x150 Pixel gesized (vorher 4608x3456)
            resized = cv2.resize(gray,(img_size,img_size)) 
            # fügt leerer Data-list jedes resized Image zu 
            data_list.append(resized)
            label_list.append(label_dict[category])
        
        except Exception as e:
            print("Exception: ",e)

# Überprüfen der zwei erstellten Listen auf denen das gesamte folgende Training
# basiert 
print('Anzahl aller Trainingsbilder:',len(data_list)) 
print('Anzahl der zugehörigen Labels:',len(label_list)) 

# Daten werden normalisiert 
data_list = np.array(data_list)/255.0  
# data_list wird in 4-Dimensionale Form gebracht, es wird dem Tensor eine 1
# hinzugefügt                                           
data_list = np.reshape(data_list,(data_list.shape[0],img_size,img_size,1))
# Labels werden ebenfalls in np-Form gebracht 
label_list = np.array(label_list)   
# Kodierung der Labels in zwei.dim. Matrix von eindim. Array 
new_label_list = np_utils.to_categorical(label_list)

np.save('data',data_list)                              
np.save('label',new_label_list)    


# =============================================================================
# Step 4: Convolutional Neural Network modellieren
# =============================================================================


# Laden der abgespeicherten Daten
data = np.load('data.npy') 
label = np.load('label.npy') 
# Sequential ist eine Art Grundgerüst
model = Sequential() 

# Adding des ersten Layers mit 200 Neuronen, einem 3x3 Faltungsfilter (faltet
# nach dem Prinzip des inneren Produkts), und dem Input (analog zu shape der
# der Daten -> 150,150,1)
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
# Aktivierungsfunktion, die das Feuerungsprinzip zwischen den Layers modelliert
# wird nach Faltung angewendet 
model.add(Activation('relu'))
# Erste Pooling-Layer mit 2x2 "Vereinfachungs"-Filter -> Verringert die Granularität
# der Auflösung und reduziert somit die zu verarbeitenden Informationen 
model.add(MaxPooling2D(pool_size=(2,2)))    
# Analog - 100 Neuronen, 3x3 Faltungsfilter
model.add(Conv2D(100,(3,3)))
# Analog
model.add(Activation('relu'))
# Analog
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten führt die vorher voneinander unabhängig verlaufenden Matrizen der 
# Teilbereiche des Bildes wieder zusammen (fully connected) und ordnet sie zu einem
# AnzahlAllerBildpunkte*1 Vektor an
model.add(Flatten())   
# Regulierungsmaßnahme gegen Overfitting, Argument bestimmt auf welche Neuronen
# herausgenommen werden 
model.add(Dropout(0.5))
# geflatteter, eindimensionaler Vektor wird in Dense-Layer von 50 Neuronen über-
# führt -> fully connected softmax
model.add(Dense(50,activation='relu'))
# letzter Layer besitzt soviele Neuronen wie es Lösungs-/ Klassifikationsmöglichkeiten
# gibt -> fully connnected softmax
# Kategorie mit höchster W.keit wird als Antwort ausgegeben (hier: 0 oder 1)
model.add(Dense(2,activation='softmax'))

# Konfigurationen für Das Training des Modells -> siehe Help
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
# Zusammenfassung der Modellarchitektur  
model.summary() 


# =============================================================================
# Step 5: Split the data_list and target and fit into the model:
# =============================================================================


# Splitten der Daten in Trainings- und Testdaten (9 zu 1)
train_data, test_data, train_label, test_label = \
    train_test_split(data, label, test_size=0.1)  

# filepath = 'model-{epoch:03d}.model'

# Das einzige beste Modell wird gespeichert 
filepath = 'model.model'

# Einstellungen für Modelltraining
# monitor -> Auf welchen Kennwerten liegt der Fokus 
# verbose -> Was wird während Training in Konsole angezeigt
# mode -> Entscheidung, welches Modell das Beste ist, orientiert sich an dem  
# Kriterium, das unter monitor genannt ist 
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', \
    verbose = 0, save_best_only = True, mode='auto')
# Erzeugt Liste der Zwischenspeicherungen 
callbacks_list = [checkpoint]
# Modell wird trainiert 
model.fit(train_data, train_label, epochs = 20, callbacks = \
    callbacks_list, validation_split = 0.2)
# Zeigt letztlich die Metrics für die Güte des Modells für die Daten im Argument 
print(model.evaluate(test_data, test_label))


# =============================================================================
# TF-LITE Konvertierung 
# =============================================================================


# Konvertiere das bereits tranierten, ursprüngliche Modell entsprechend der
# tf-lite-Konvertierung 
converter = tf.lite.TFLiteConverter.from_saved_model('model.model')
tflite_model = converter.convert()
# Speichere das konvertierte Modell
open("converted_model.tflite", "wb").write(tflite_model)

    
# =============================================================================
# Step 6: Bestes Modell in Real Life mit Webcam benutzen 
# =============================================================================


# Hier noch einmal definieren für Trennung des Codes
img_size = 150
# Modell laden
# du musst das Modell für diesen Ordner nochmal neu tranieren wahrscheinlich 
loaded_model = load_model(r'C:\YourPath\model.model')
# Classifier, der direkt nach Gesichtern in gegebenen Bild sucht 
faceCascade=cv2.CascadeClassifier \
    (r'C:\YourPath\haarcascade_frontalface_default.xml') 
# Starte Videoaufnahme  
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
        # Grayscale auf image (?)
        face_img = gray[y:y+w,x:x+h]
        # Größe des durch Classifier ausgesuchten Bildes auf img_size mappen
        resized = cv2.resize(face_img,(img_size,img_size))
        # Pixelwert-Normalisierung in Bereich zwischen 0 und 1
        normalized = resized/255.0
        # Matrix zu np-array shapen und 1er Vektor anhängen
        reshaped = np.reshape(normalized,(1,img_size,img_size,1)) 
        # Resaped image wird an Modell gegeben 
        result = loaded_model.predict(reshaped) 
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


