import cv2 
from ultralytics import YOLO 
import time 
import numpy as np
#model = YOLO('yolov8m.pt')
model2 = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

def get_bounding_box(frame):
    model = YOLO('yolov8n.pt')

    # Capture a frame

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img, device = 'cpu')
    x1 = 0 
    y1 = 0 
    x2 = 0 
    y2 = 0 
    # Find the bounding box coordinates
    r = results[0]
    boxes = r.boxes
    for box in boxes:
        c = box.cls
        if model.names[int(c)] == 'person':
            b = box.xyxy[0]  # Get the bounding box coordinates
            x1 = int(b[0]) #x1
            y1 = int(b[1]) #y1
            x2 = int(b[2]) #x2
            y2 = int(b[3]) #y2

    # Return the bounding box coordinates
    return x1, y1, x2, y2

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
<<<<<<< HEAD
    num = 0 
    x = kp_coordinates[num][0] #nose
    y = kp_coordinates[num][1]
    cv2.circle(frame, (int(x), int(y)), 5, (255,0,0), thickness=5)

=======
    x = kp_coordinates[1][0]
    y = kp_coordinates[1][1]
    print(x)
    print(y)
    x1, y1, x2, y2 = get_bounding_box(frame)
    roi = frame[y1:y2, x1:x2]
    cv2.circle(roi, (int(x), int(y)), 5, (255,0,0), thickness=5)
    cv2.imshow('roi', roi)
>>>>>>> da66962bc47d02bd42900958f85457bea5d3f8dc
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()