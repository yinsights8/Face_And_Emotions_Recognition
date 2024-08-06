import cv2
from face_detection import RetinaFace

import cvzone as cv

detector = RetinaFace(gpu_id=-1)
image = "E:/My_Model/4. Computer Vision Games/ArchFace/ArchFaceRecognition/dataset/jenna_ortega/1.jpg"
cam = cv2.VideoCapture(0)

while cam.isOpened():
    succ, frame = cam.read()
    if not succ:
        break

    results = detector.detect(frame)
    # print(results[0][0])
    if len(results) != 0:
        x, y, w, h = results[0][0]
        # x1, y1 = x + w, y + h
        # x, y, x1, y1 = int(x), int(y), int(x1), int(y1)

        cv.cornerRect(img=frame, bbox=(int(x), int(y), int(w-x), int(h-y)), l=20, t=3, colorR=(255, 0, 255))

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
