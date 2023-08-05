import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)
while True:
    
    ret, frame = cap.read()
    start_time = time.time()
    # Run YOLOv8 inference on the frame
    output = model(frame, save=False)
    # Extract the pose tensor from the output using the following line:
    pose_tensor = output[:, model.model.names.index('pose')]
    # Extract the key-points data from pose_tensor using the following line:
    keypoint_data = pose_tensor[0].cpu().detach().numpy()
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print("FPS :", fps)
    
    # Draw keypoints on the frame
    for x, y in keypoint_data:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    cv2.putText(frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
