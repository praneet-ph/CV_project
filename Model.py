#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import keras
from PIL import Image
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import numpy as np
import tables
import random


# In[22]:


#build a dictionary to save the gesture name information
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('C:/CS/C_vision/project/dataset/Dataset_Binary/Dataset_Binary'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1


# In[23]:


lookup


# In[24]:


#build a function to load the data
def get_data():
    imagecount = 0 
    xy_data = []
    #go over all the gesture folders
    for i in os.listdir('C:/CS/C_vision/project/dataset/Dataset_Binary/Dataset_Binary/'):
        #get rid of hidden folder
        if not i.startswith('.'): 
            count = 0
            #go over the all the images in folder
            for j in os.listdir('C:/CS/C_vision/project/dataset/Dataset_Binary/Dataset_Binary/' + i + '/'):
                # read the image and transfer into greyscale image
                img = Image.open('C:/CS/C_vision/project/dataset/Dataset_Binary/Dataset_Binary/' + i + '/' + j).convert('L')                
                #resize the image to fit the foramt of VGG
                img = img.resize((224, 224))
                arr = np.array(img)
                xy_data.append([arr,lookup[i]]) 
            imagecount = imagecount + count            
    return xy_data


# In[25]:


#build a function to process the data 
def process_data(x, y):
    x = np.array(x, dtype = 'float32')
    x = x.reshape((len(x), 224, 224, 1))
    x /= 255
    y = np.array(y)
    y = to_categorical(y)
    return x, y


# In[26]:


#load data
XY_data = get_data()


# In[27]:


#random the order of the data 
random.shuffle(XY_data)


# In[28]:


#spilt the image content and gesture inforamtion
X_data = []
Y_data = []
for i in XY_data:
    X_data.append(i[0])
    Y_data.append(i[1])


# In[29]:


#define the rate of train and test 
rate = int(len(X_data)*0.8)


# In[30]:


#split the train and test part 
X_train = X_data[:rate]
X_test = X_data[rate:]
Y_train = Y_data[:rate]
Y_test = Y_data[rate:]


# In[31]:


#process the data
X_train, Y_train = process_data(X_train, Y_train)
X_test, Y_test = process_data(X_test, Y_test)


# In[32]:


#Normal Neural network part


# In[33]:


#Build the network
from keras import models, layers
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dropout(0.25, seed=21))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(15, activation='softmax'))


# In[34]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[36]:


model.fit(X_train, Y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, Y_test))


# In[37]:


#data augmentation with rotation/shift/flip
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=64),
                    steps_per_epoch=len(X_train) / 64, epochs=10, validation_data=(X_test, Y_test))


# In[38]:


#get the summary of the model
model.summary()


# In[39]:


# calculate the accuracy rate on the test data
[loss, acc] = model.evaluate(X_test,Y_test,verbose=1)
print("Accuracy:" + str(acc))


# In[40]:


# make predcitions on test data
predictions = np.argmax(model.predict(X_test), axis=-1)


# In[41]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# print the confusion matrix
Y_test1=np.argmax(Y_test, axis=1)
confusion_matrix(Y_test1, predictions)
for i in confusion_matrix(Y_test1, predictions):
    print (i)


# In[ ]:


# Save the model file
model.save('model.h5')


# In[ ]:





# In[ ]:


#VGG part


# In[12]:


#build a function to process the vgg data
def process_data_VGG(x, y):
    x = np.array(x, dtype = 'float32')
    x = np.stack((x,)*3, axis=-1)
    x = x.reshape((len(x), 224, 224, 3))
    x /= 255
    
    y = np.array(y)
    y = to_categorical(y)
    return x, y


# In[11]:


X_train_VGG = X_data[:rate]
X_test_VGG = X_data[rate:]
Y_train_VGG = Y_data[:rate]
Y_test_VGG = Y_data[rate:]


# In[13]:


#process the vgg data
X_train_VGG, Y_train_VGG = process_data_VGG(X_train_VGG, Y_train_VGG)
X_test_VGG, Y_test_VGG = process_data_VGG(X_test_VGG, Y_test_VGG)


# In[15]:


#get the network of VGG from imagenet
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#define the optimizer
optimizer1 = optimizers.Adam() 


# In[16]:


#define the callback
class MetricsCheckpoint(Callback):
    #Callback that saves metrics after each epoch
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
        
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


# In[17]:


# Add VGG network
base_model = model1  
x = base_model.output
#add layers
x = Flatten()(x)
x = Dropout(0.5)(x)

predictions = Dense(15, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# use the vgg network instead of train it
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

model.summary()

model.fit(X_train_VGG, Y_train_VGG, epochs=10, batch_size=64, validation_data=(X_test_VGG, Y_test_VGG),
          verbose=1, callbacks=[MetricsCheckpoint('logs')])


# In[18]:


#save the VGG model
model.save('Model_VGG.h5')


# In[19]:


#calculate the accuracy rate
[loss, acc] = model.evaluate(X_test_VGG,Y_test_VGG,verbose=1)
print("Accuracy:" + str(acc))


# In[20]:


#calculate the confusion matrix
predictions = np.argmax(model.predict(X_test_VGG), axis=-1)
Y_test_VGG1=np.argmax(Y_test_VGG, axis=1)
confusion_matrix(Y_test_VGG1, predictions)
for i in confusion_matrix(Y_test_VGG1, predictions):
    print (i)


# In[ ]:


#References:
    #Keras Models   https://www.tensorflow.org/api_docs/python/tf/keras/Model
    #Open CV2 Image processing    https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
    #Image fit VGG format    https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
    #                        https://numpy.org/doc/stable/reference/generated/numpy.stack.html

