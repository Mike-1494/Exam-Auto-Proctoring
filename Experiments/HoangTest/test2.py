import cv2
from ultralytics import YOLO
import time
import imageio
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
class PosePredictor(DetectionPredictor):
    
    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                      self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results1 = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            if len(pred) == 0:
                pred_kpts = None
            else:
                pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape)
                pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)

            path, _, _, _= self.batch
            img_path = path[i] if isinstance(path, list) else path
            results1.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))

            if pred_kpts is not None:
                for idx, kpt in enumerate(pred_kpts[0]):
                    print(f"Keypoint {idx}: ({kpt[0]:.2f}, {kpt[1]:.2f})")
        return results1

#Load the Yolov8 model
model = YOLO('yolov8n-pose.pt')
#open cam
cap = cv2.VideoCapture(0)
#creat a pose predictor object
predictor = PosePredictor(overrides=dict(model='yolov8n-pose.pt'))

while True:
    ret,frame = cap.read()
    start_time = time.time()
    # Run pose detection on the frame
    results1 = predictor(frame)
    
    # Visualize the results on the frame
    annotated_frame = results1[0].plot()
    
    # print keypoints index number and x,y coordinates
    for idx, kpt in enumerate(results1[0].keypoints[0]):
        
        print(f"Keypoint {idx}: ({int(kpt[0])}, {int(kpt[1])})")
        annotated_frame = cv2.putText(annotated_frame, f"{idx}:({int(kpt[0])}, {int(kpt[1])})", (int(kpt[0]), int(kpt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print("FPS :", fps)
    
    cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
    
    # Display the annotated frame
    cv2.imshow("Pose Detection", annotated_frame)
    
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    