#!/usr/bin/env python
# coding: utf-8

# In[33]:


import cv2
import numpy as np
from keras.models import load_model
import time
import copy

# the dictionary that save the information of the gestures 
gesture_names = {0:'down',
                 1:'eight',
                 2:'five',
                 3:'four',
                 4:'left',
                 5:'nine',
                 6:'one',
                 7:'right',
                 8:'seven',
                 9:'six',
                 10:'stop',
                 11:'three',
                 12:'two',
                 13:'up',
                 14:'zero' }


#normal NN(!!!change if use this model!!!)
#model = load_model('C:/Users/dyson/model.h5')
#VGG
model = load_model('C:/Users/dyson/Model_VGG.h5')

#function to predict the image with normal neural network
def predict_rgb_image(img):
    pred_array = model.predict(img)
    result = gesture_names[np.argmax(pred_array[0], axis=-1)]
    #result = gesture_names[model.predict_classes(img)[0]]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return (result, score)

#function to predict the image with VGG network
def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score


# parameters
prediction = ''
action = ''
score = 0
img_counter = 500
#settings
#width of area of interest
aoi_width = 0.5  
#height of area of interest
aoi_height = 0.6  
# binary threshold
threshold = 62 
# GaussianBlur parameter
blurValue = 41 
#Background  threshold
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

#function to remove the background
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


#setting camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
    ret, frame = camera.read()
    # smoothing filter
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  
    # flip the frame horizontally to make the video have the same order with real world(like a mirror)
    frame = cv2.flip(frame, 1) 
    #draw area of interest
    cv2.rectangle(frame, (int(aoi_width * frame.shape[1]), 0),
                  (frame.shape[1], int(aoi_height * frame.shape[0])), (0, 255, 0), 2)
    #show the video with area of interest
    cv2.imshow('WebCam', frame)

    # if the background is captured,run this part
    if isBgCaptured == 1:
        #remove the background
        img = remove_background(frame)
        #get area of interest
        img = img[0:int(aoi_width * frame.shape[0]),
              int(aoi_height * frame.shape[1]):frame.shape[1]]
        
        # convert the image into greyscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #show the greyscale video
        cv2.imshow('gray', gray)
        #apply the Gaussianblur
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #show the blured video
        cv2.imshow('blur', blur)
        #apply the threshold
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        #Normal NN(!!!change if use this model!!!)
        #target = cv2.resize(thresh, (224, 224))
        #target = target.reshape(1, 224, 224, 1)
        #VGG : make prediction with VGG model
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction,score = predict_rgb_image_vgg(target)
        #show result with text
        cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
        #Show the result video
        cv2.imshow('Binary', thresh)
        
        #copy the threshold iamges
        thresh1 = copy.deepcopy(thresh)
        #find contours
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            # find the biggest contour
            for i in range(length):  
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            #draw the contours
            cv2.drawContours(drawing, [res], 0, (0, 0, 255), 2)
            cv2.drawContours(drawing, [hull], 0, (255, 0, 0), 3)
        #show the video of contours
        cv2.imshow('Contour', drawing)

        

    # press ESC to exit 
    k = cv2.waitKey(10)
    if k == 27:  
        cv2.destroyAllWindows()
        break
        
    # press 'b' to capture the background    
    elif k == ord('b'): 
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
        


# In[ ]:


#References:
    #Keras Models   https://www.tensorflow.org/api_docs/python/tf/keras/Model
    #Open CV2 Camera    https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    #Open CV2 Image processing    https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

