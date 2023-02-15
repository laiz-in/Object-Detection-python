import numpy as np
import cv2
#import numpy and opencv

#define the haarcascade frontal face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#open the webcam ,0 is the device number
cap = cv2.VideoCapture(0)


cap.set(3,640) # set Width
cap.set(4,480) # set Height

#condition gets true if the camera opens
while True:
    #catching hte frames
    ret, img = cap.read()

    #converting to grey scale image
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #detetcting face in the frame
    faces = faceCascade.detectMultiScale(
        gray,#name of the image
        scaleFactor=1.3, #scale factor usually between 1.0 to 1.5
        minNeighbors=1, #minimum no of faces to detect, in our case its one    
        minSize=(20, 20 )#minimum size of the onject
    )

    #faces will return x,y,w,h cordinates , below "for loop" will draw the borders using rthe cordinates
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    
    #show the webcam with dtetected face
    cv2.imshow('video',img)

    #breaking the window if esc presses
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#releasing the webcam
cap.release()
cv2.destroyAllWindows()