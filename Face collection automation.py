#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np



# Loading HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    faces = face_classifier.detectMultiScale(img, 1.7, 5)
    
    if faces is ():
        return None
    
    # Cropping to desired dimensions.
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face


# In[3]:


cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory 
        file_name_path = './Images/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("No Face Found")
        pass

    if cv2.waitKey(1) == 13 or count == 250:
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")





