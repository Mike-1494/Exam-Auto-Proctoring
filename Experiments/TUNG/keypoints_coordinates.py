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
    frame = cv2.imread('D:\Git\Exam-Auto-Proctoring\Experiments\TUNG\multyperson.jpg') 
    
    #bounding_boxes = model(frame)
    #print(bounding_boxes)
    #result = model(frame, save = False)q
    result2 = model2(frame, save = False, device = 'cpu')
    print(result)
    
    keypoints = result2[0].keypoints
    keypoints_np = keypoints.numpy()
    #print(keypoints_np)
    kp_coordinates = keypoints_np.xy[0]
    #print(keypoints_np.shape)
    num = 0 
    x = kp_coordinates[num][0] #nose
    y = kp_coordinates[num][1]
    cv2.circle(frame, (int(x), int(y)), 5, (255,0,0), thickness=5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()