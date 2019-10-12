import cv2
import numpy as np

img=np.zeros((512,512,3),np.int8)

#Function to draw a circle
def draw_circle(event,x,y,flags,param):
    if(event==cv2.EVENT_LBUTTONDOWN):
        cv2.circle(img,(x,y),100,(255,0,255),-1)
        
cv2.namedWindow('my_drawing')
cv2.setMouseCallback('my_drawing',draw_circle)

#Display the image
while True:
    cv2.imshow('my_drawing',img)
    if cv2.waitKey(20) & 0xFF==27:
        break
cv2.destroyAllWindows()
        