import cv2
import numpy as np
import matplotlib.pyplot as plt

# img1=cv2.imread('penguin.jpeg')
# img2=cv2.imread('space.jpeg')
# img3=cv2.imread('illuminati.jpeg')

# #applying a threshold

# rows,columns,channels=img3.shape
# roi=img2[0:rows,0:columns]

# img3gray=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret, mask=cv2.threshold(img3gray,180,255,cv2.THRESH_BINARY_INV)

# mask_inv=cv2.bitwise_not(mask) #low lvl logical op

# img2_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)
# img3_fg=cv2.bitwise_and(img3,img3,mask=mask)

# dst=cv2.add(img2_bg,img3_fg)
# img1[0:rows,0:columns]=dst


# cv2.imshow('mask',mask)
# cv2.imshow('mask_inv',mask_inv)
# cv2.imshow('img3_fg',img3_fg)
# cv2.imshow('img2_bg',img2_bg)
# cv2.imshow('final',img1)

cap=cv2.VideoCapture(0)

frontalFace_casc=cv2.CascadeClassifier("haarcascade_frontalface.xml")
profileFace_casc=cv2.CascadeClassifier("haarcascade_profileface.xml")
while True:
    ret, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Pfaces=profileFace_casc.detectMultiScale(gray,1.3,5)
    Ffaces=frontalFace_casc.detectMultiScale(gray,1.3,5)
    for(px,py,pw,ph) in Pfaces:
        cv2.rectangle(img,(px,py),(px+pw,py+ph),(255,0,0),2)
    for(fx,fy,fw,fh) in Ffaces:
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()