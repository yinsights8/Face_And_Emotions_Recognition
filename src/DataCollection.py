import os

import pandas as pd

from src import config
from src.faceDetectors.SCRFD_FaceDetector.scrfdDetector import SCRFD
import logging
import cv2
import numpy as np
import cvzone as cvz
import time
from src.face_align import norm_crop
import datetime

# arguments for face detection
no_of_faces = config.no_of_faces
save_dir = config.unTrained_data_dir
detection_model = config.scrfd_weights
embeddings_file = config.feature_embs


class CollectDataFromCamera:
    # print("Collecting Data.........")
    def __init__(self, personName, FaceNums=no_of_faces, saveDir=save_dir,
                 detectionModel=detection_model, ctx_id=0):
        self.personName = personName.lower()
        self.saveDir = saveDir
        self.FaceNums = FaceNums
        self.detectionModel = detectionModel
        self.target_size = (112, 112)

        # Initialize FPS
        self.cTime = 0
        self.pTime = 0

        # initialize Detector
        # self.detector = MTCNN()
        self.detector = SCRFD(model_file=self.detectionModel)
        # self.detector.prepare(ctx_id=ctx_id)
        logging.info(f"INFO[{__name__}] SCRFD Detector Initialized")

        # directory to save images
        self.save_img_to_dir = os.path.join(self.saveDir, self.personName)


        # check if the person's name is already exists in the database


    def getDataFromCamera(self, camNo=0):
        cap = cv2.VideoCapture(camNo)
        logging.info(f"INFO[{__name__}] Video mode Initialized")

        faces = 0
        frames = 0
        max_faces = self.FaceNums
        max_bbox = np.zeros(4)


        if not (os.path.exists(self.save_img_to_dir)):
            logging.info(f"INFO[{__name__}] {self.save_img_to_dir} already present...")
            os.makedirs(self.save_img_to_dir)
        else:
            logging.info(f"INFO[{__name__}] {self.personName} already present...")
            return f"{self.personName} already present..."

        while faces < max_faces:
            ret, frame = cap.read()
            frames += 1

            # Get all faces on current frame
            results = self.detector.detect(frame, (640, 480))

            if len(results) != 0:
                boxes, landmarks = results
                # Get only the biggest face
                max_area = 0
                for i in range(len(boxes)):
                    x1, y1, x2, y2, _ = boxes[i]
                    bbox = np.array([x1, y1, x2, y2])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_bbox = bbox
                        max_landmarks = landmarks[i]  # Save the corresponding landmarks
                        max_area = area

                max_bbox = max_bbox[0:4]

                # get each of 3 frames
                if frames % 3 == 0:
                    x1, y1, x2, y2 = max_bbox
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Convert landmarks to the required format
                    max_landmarks = np.array(max_landmarks).reshape((2, 5)).T

                    # Use norm_crop to crop the face image
                    nimg = norm_crop(frame, max_landmarks, image_size=112)
                    save_img = os.path.join(f"{self.saveDir}/{self.personName}", "{0}_{1}.jpg".format(self.personName, faces))
                    cv2.imwrite(save_img, nimg)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
                logging.info(f"INFO[{__name__}] Images are captured...")

            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def getFaceFromCam(self, camNo=0):
        print("Preparing Embeddings....")
        embs_file = os.path.join(embeddings_file, "Embeddings.csv")
        if os.path.exists(embs_file):
            embed_file = pd.read_csv(embs_file)
            if self.personName in embed_file['image_name'].tolist():
                logging.info(f"INFO[{__name__}] Person Already exists in the DataBase, need SignIp only")
                print(f"{self.personName} Already Exists !!")
            else:
                print("Capturing Data......")
                self.getDataFromCamera(camNo)
