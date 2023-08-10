from ultralytics import YOLO 
import cv2
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('"D:\Git\Exam-Auto-Proctoring\Experiments\HoangTest\dance.mp4"')
fps = cap.get(cv2.CAP_PROP_FPS)
count = 0

while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('Frame', frame)
    if count % fps == 0:
        cv2.imwrite(f'image_{count}.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()