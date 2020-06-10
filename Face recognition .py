#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras 
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input 
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob


# In[6]:


IMAGE_SIZE= [224, 224]

train_path= "Images/train"
test_path= "Images/test"

#since we are using a color image a third layer is attached(+[3])
#weights of imagenet is used.
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



for layer in vgg.layers:
    layer.trainable = False # we are using a pre-trained model. Layers need not be trained.
  

  
#getting number of classes
folders = glob('Images/train/*')
  


x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x) #len(folder) defines number of output neurons

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[7]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('Images/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Images/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[8]:


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs= 5,
  verbose=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[10]:


import matplotlib.pyplot as plt

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()#show plot




import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_VGG16_model.h5')





