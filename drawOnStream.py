import cv2

#Callback Function
def draw_rect(event,x,y,flags,params):
    global p1,p2,topLeftClicked,bottomRightClicked
    
    if event==cv2.EVENT_LBUTTONDOWN:
        
    
        if topLeftClicked and bottomRightClicked:
            p1=(0,0)
            p2=(0,0)
            topLeftClicked=False
            bottomRightClicked=False

        if topLeftClicked==False:
            p1=(x,y)
            topLeftClicked=True
        elif bottomRightClicked==False:
            p2=(x,y)
            bottomRightClicked=True

#Global variables
p1=(0,0)
p2=(0,0)
topLeftClicked=False
bottomRightClicked=False

#Dock the callback 
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw_rect)

#Access the video
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    
    if topLeftClicked:
        cv2.circle(frame,center=p1,radius=5,color=(0,0,255),thickness=-1)
    if topLeftClicked & bottomRightClicked:
        cv2.rectangle(frame,p1,p2,(0,0,255),3)
        
    cv2.imshow('frame',frame)
    
    
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()