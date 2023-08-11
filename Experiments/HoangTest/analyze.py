from ultralytics import YOLO 
import cv2
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('D:\Git\Exam-Auto-Proctoring\Experiments\HoangTest\dance.mp4')
count = 0

while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('Frame', frame)
    cv2.imwrite(f'D:\Git\Exam-Auto-Proctoring\Experiments\HoangTest\Image\image_{count}.jpg', frame) 
    count += 1 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()