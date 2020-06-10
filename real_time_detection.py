#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np


# In[2]:


from keras.preprocessing import image

model = load_model('facefeatures_VGG16_model.h5')#saved model is loaded here

#loading frontalface cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[3]:


def face_extractor(img):

    crop= []
    faces = face_cascade.detectMultiScale(img, 1.7, 5)
    
    if faces is ():
        return None

    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cropped_face = img[y:y+h, x:x+w]
            crop.append(cropped_face)



        return crop


# In[4]:


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    
    face = face_extractor(frame)
    if type(face) is np.ndarray: #checking for the image value encoded in numpmy array
        face = cv2.resize(face, (224,224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis = 0)
        pred = model.predict(img_array)
        print(pred)
        
        name = "No Match Found"
        
        #defining a threshold for each class/person for prediction
        if (pred[0][0]>0.5):
            name = "Sam"
        if (pred[0][1]>0.5):
            name = "Hitesh"
        
        cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No Face Found", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




