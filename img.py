import cv2    
import time
cpt = 0


maxFrames = 50


cap=cv2.VideoCapture('IMG-4552.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1080,500))
    time.sleep(0.01)
    frame=cv2.flip(frame,1)
    cv2.imshow("test window", frame)
    cv2.imwrite(r"C:\Users\manga\Downloads\yolov8-custom-object-training-tracking-main\yolov8-custom-object-training-tracking-main\images-cstm\surfing_%d.jpg" %cpt, frame)
    cpt += 1
    if cv2.waitKey(5)&0xFF==27:
        break
cap.release()   
cv2.destroyAllWindows()
