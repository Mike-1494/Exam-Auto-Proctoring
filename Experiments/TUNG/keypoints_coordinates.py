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
    frame = cv2.imread('D:\Exam-Cheating-Detection\Experiments\TUNG\jisoo_hit.jpg') 
    
    #bounding_boxes = model(frame)
    #print(bounding_boxes)
    #result = model(frame, save = False)q
    result2 = model2(frame, save = False, device = 'cpu')
    keypoints = result2[0].keypoints
    keypoints_np = keypoints.numpy()
    #print(keypoints_np)
    '''
    keypoints_np_reduce = np.squeeze(keypoints_np)
    print(keypoints_np.shape)
    print(keypoints_np_reduce.shape)
    print("in ra toa do cac keypoint")
    for person in keypoints_np:
        for keypoint in person:
            x, y, confidence = keypoint
            if confidence > 0.3: 
                print(f"Keypoint: X={x}, Y={y}")
    #frame = result[0].plot()
    frame = result2[0].plot()
    '''
    kp_coordinates = keypoints_np.xy[0]
    #print(keypoints_np.shape)
    x = kp_coordinates[1][0]
    y = kp_coordinates[1][1]
    print(x)
    print(y)
    x1, y1, x2, y2 = get_bounding_box(frame)
    roi = frame[y1:y2, x1:x2]
    cv2.circle(roi, (int(x), int(y)), 5, (255,0,0), thickness=5)
    cv2.imshow('roi', roi)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()