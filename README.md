# Embedded CNN Mask Detector for Smart Lock Control

## Objective 
- Train a CNN on dataset with images of faces wearing or not wearing a mask
- Compress the model for edge device deploymet (Tensorflow Lite)
- Upload it to a Rasppery Pi connected to a camera and an electric lock
- Let the compressed model scan in real time faces exposed to the camera and open the lock if fase wearing a mask is detected 

## Hierarchy of actions
1. MaskDetectorTraining.py
2. MaskDetector.py - run model
3. MaskDetectorLite.py - run compressed model 


