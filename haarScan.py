import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import os


cap=cv2.VideoCapture(0)

classifier =load_model('assets\model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

frontalFace_casc=cv2.CascadeClassifier("assets\haarcascade_frontalface.xml")
profileFace_casc=cv2.CascadeClassifier("assets\haarcascade_profileface.xml")

pflag=0
fflag=0
while True:
    ret, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Pfaces=profileFace_casc.detectMultiScale(gray,1.3,5)
    Ffaces=frontalFace_casc.detectMultiScale(gray,1.3,5)
    for(px,py,pw,ph) in Pfaces:
        pflag=1

        cv2.rectangle(gray,(px,py),(px+pw,py+ph),(255,0,0),2)

        roi_gray = gray[py:py+ph,px:px+pw]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (px,py)
            cv2.putText(gray,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
    else:
        for(fx,fy,fw,fh) in Ffaces:
            cv2.rectangle(gray,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)

            roi_gray = gray[fy:fy+fh,fx:fx+fw]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (fx,fy)
                cv2.putText(gray,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 

    cv2.imshow('footage',gray)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()