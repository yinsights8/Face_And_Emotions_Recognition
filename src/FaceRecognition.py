import logging
import time

import numpy as np

from src.FaceModel.ResNet_Arch import resnet_inference
from src.faceDetectors.SCRFD_FaceDetector.scrfdDetector import SCRFD
from src.FaceModel.FaceEmbeddingLoader import FaceEmbeddingLoader
from src.face_align import norm_crop
from src import config
import torch
import os
from src.tracker.sort import Sort
# from deep_sort_realtime.deepsort_tracker import DeepSort
import cvzone as cvz
import cv2

# 1. import all necessary paths
detectorPath = config.scrfd_weights
backbone_faceModel = config.face_model_backbone
faceRecModel = config.face_model_path
faceEmbeddings_path = config.feature_embs


# 2. preprare recongnition software
class FaceIdentity:
    def __init__(
            self, detectionModel=detectorPath, cxt_id=0,
            modelBackbone=backbone_faceModel,
            faceRecModel=faceRecModel, faceEmbs_path=faceEmbeddings_path):

        # Hardware Configuration '0 for Cuda and -1 for CPU'
        self.cxt_id = cxt_id

        # face detection model
        self.detectionModel = detectionModel

        # archFace supported model 'ResNet'
        self.modelBackbone = modelBackbone

        # ArchFace model 'for face embedding'
        self.ArchFaceModel = faceRecModel

        # Load Face embeddings
        self.faceEmbs_path = faceEmbs_path

        # Initialize FPS
        self.cTime = 0
        self.pTime = 0

        # set cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {0} is ready...".format(self.device))

        # set face detector
        self.detector = SCRFD(model_file=self.detectionModel)
        self.detector.prepare(ctx_id=self.cxt_id)
        logging.info("Face detection model is ready...")

        # face recognition model to create embeddings
        self.FaceRecognition = resnet_inference(
            model_name=self.modelBackbone,
            pretrained_model=self.ArchFaceModel
        )
        logging.info("Face Recognition model is ready...")

        # load Face embeddings
        self.featureEmbs = FaceEmbeddingLoader()
        self.pre_image_name, self.pre_image_embs = self.featureEmbs.load_embeddings()
        logging.info("Embeddings loadded successfully...")

        # deepsort tracker
        self.tracker = Sort(max_age=20, iou_threshold=0.3)

    def GetSimilariyScoreAndIndex(self, AlignedFaces):
        """
         Convert the detected face into embeddings.

        Args:
            AlignedFaces: an AlignedFaces input face image from Camera.

        Returns:
            tuple: A tuple containing the recognition score and person identity.
        """

        # get the embeddings of a person from Front camera
        self.AlignedFaces = self.featureEmbs.getEmbeddings(AlignedFaces)
        logging.info("collecting front embeddings from camera...")

        # comapair the embeddings that are already present in database
        score, index = self.featureEmbs.CompairEmbeddings(self.AlignedFaces, self.pre_image_embs)
        identity = self.pre_image_name[index]
        logging.info("Collected and calculated Cosine similarity...")
        return score, identity

    def RealTimeRecognition(self, FaceFromCamera):
        """
        Detect and Identify the Person from Camera

        :param Frame: Input Frames from Real Time Camera
        :return: numpy.ndarray: The frame with recognized faces and their names annotated.
        """

        caption = ""
        NameColor = (0, 255, 0)
        RectColor = (0, 255, 0)
        detections = np.empty((0, 5))

        # 1. detect face in the camera
        results = self.detector.detect(FaceFromCamera, input_size=(640, 480))
        if len(results) != 0:
            boxes, landmarks = results

            # ------------------------------------------------Detection Tracker------------------------------------------------

            for i in range(len(boxes)):

                # 2. align the faces
                aligned_face = norm_crop(img=FaceFromCamera, landmark=landmarks[i])
                logging.info("Aligned the Faces")

                # 3. get the similarity score and their names
                cosineScore, Name = self.GetSimilariyScoreAndIndex(AlignedFaces=aligned_face)
                logging.info("Calculated the Cosine Score and Got the Name")

                if Name is not None:
                    if cosineScore < 0.25:
                        caption = "UnKnown"
                        NameColor = (0, 29, 255)  # Red light color
                        RectColor = (0, 29, 255)  # Red light color
                        logging.info("Unknown Entity Detected")
                    else:
                        caption = f"{Name} : {cosineScore :.2f}"
                        NameColor = (0, 255, 0)  # green light color
                        RectColor = (0, 255, 0)  # green light color
                        logging.info("Known Entity Detected")

                    # print(f"Identity: {caption}, Confidence:{cosineScore}")
                    x, y, w, h, cnf = boxes[i]

                    CurrentArray = np.array([x, y, w, h, cnf])
                    detections = np.vstack((CurrentArray, detections))

            TrackerResults = self.tracker.update(detections)

            for result in TrackerResults:
                x1, y1, w, h, id = result
                x2, y2 = w - x1, h - y1
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

                # ------------------------------------------------Detection Tracker------------------------------------------------

                # Calculte FPS
                self.cTime = time.time()
                fps = 1 / (self.cTime - self.pTime)
                self.pTime = self.cTime

                cvz.putTextRect(FaceFromCamera, f"FPS: {int(fps)}", (50, 50), 1, 2, colorT=NameColor)

                cvz.cornerRect(img=FaceFromCamera, bbox=(int(x1), int(y1), int(x2), int(y2)), l=20, t=3, colorR=RectColor)

                text = f"{int(id)}: {caption}"
                # cv2.putText(FaceFromCamera, text=text, org=(max(35, x1), max(35, y1 - 15)), fontFace=1, color=NameColor)
                cvz.putTextRect(FaceFromCamera, text=text, pos=(max(35, int(x1)), max(35, int(y1) - 15)), scale=0.8,
                                thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorT=NameColor, colorR=None, colorB=None)
            # 4. return the frame

        return FaceFromCamera
