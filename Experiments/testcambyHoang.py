import cv2 
from ultralytics import YOLO 
import time 
import numpy as np
#model = YOLO('yolov8m.pt')
model2 = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0                                                                                                                               

while True:
    ret, frame = cap.read() 
    
    #bounding_boxes = model(frame)
    #print(bounding_boxes)
    #result = model(frame, save = False)q
    result2 = model2(frame, save = False, device = 'cpu')
    keypoints = result2[0].keypoints
    keypoints_np = keypoints.numpy()
    print(keypoints_np)
    x, y = keypoints_np[0][1]
    print(x)
    print(y)
    #frame = result[0].plot()
    frame = result2[0].plot()
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, str(fps), (50,50), cv2.FONT_HERSHEY_COMPLEX , 1,  (0,255,0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()