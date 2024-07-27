import winsound

from ultralytics import YOLO
import math,time,cv2,cvzone,numpy as np
from keras.models import model_from_json

        # all model loading section
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
print("Loaded Emotional model from disk")
        # yolo model
model = YOLO("../Yolo-Weights/yolov8s.pt")
print("Loaded Yolo model from disk")
        # detection content
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird","cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
RejectClass = [1,1,1,1,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,1,
              0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,1,0,0,0
              ]
#-----------------------------------------------------------------

pTime=0
cap = cv2.VideoCapture(1)
detect={
    "person":[],"knife":[]
}
invert=lambda a:"knife" if(a=="person") else "person"
def collision(obj,arr):
    x,y,w,h=(0,1,2,3)
    for a in arr:
        if( obj[x] < a[x] + a[w] and obj[x] + obj[w] > a[x] and
            obj[y] < a[y] + a[h] and obj[h] + obj[y] > a[y]):
            print("Thief..................")
            winsound.Beep(2000, 1500)

while True:
    # Image read,resize and fps
    cTime = time.time()
    success, img = cap.read()
    if not success: break
    frame = cv2.resize(img, (1280, 720))
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,3,(5, 2, 255),2)

    # emotion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if (RejectClass[cls]):
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                Detection = classNames[cls]
                if (Detection in ["person", "knife"]):
                    cur=(x1, x2, w, h)
                    detect[Detection].append(cur)
                    collision(cur,detect[invert(Detection)])

                cvzone.cornerRect(frame, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100

                if Detection=='person':
                    roi_gray_frame = gray_frame[y1:y1 + h, x1:x1 + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    strs=emotion_dict[maxindex]
                    cv2.putText(frame, emotion_dict[maxindex], (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 2, 255),2,cv2.LINE_AA)
                cvzone.putTextRect(frame, f'{Detection} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    detect["person"].clear()
    detect["knife"].clear()
    # print(fps)
    cv2.imshow('Detection', frame)
    # cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
