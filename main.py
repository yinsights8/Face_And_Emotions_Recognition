# 1. import Libraries
import logging
import torch
from src.FaceRecognition2 import FaceIdentity
from src import config
from src.faceDetectors.SCRFD_FaceDetector.scrfdDetector import SCRFD
import cv2
import cvzone as cvz
from src.image_to_face_embeddings import FaceEmbeddingPipeline
from src import image_to_face_embeddings

from src.DataCollection import CollectDataFromCamera

# 1.1 import all the configuration files
detectionModel = config.scrfd_weights
trained_data = config.Trained_data_dir
do_training_data = config.unTrained_data_dir

# 2. create an embedding instance
embeddings = FaceEmbeddingPipeline(trained_data=trained_data, do_training_data=do_training_data)

# 3. Ask For a Name from User
YourName = input("Enter Your Full Name: ")
LastName = input("Enter Your Last Name: ")
fullName = "_".join([YourName, LastName])
collectData = CollectDataFromCamera(personName=fullName)
collectData.getFaceFromCam()

# 4. generate Face embeddings from collected data
embeddings.process_and_embed_faces()

# 5. face detection model
detector = SCRFD(model_file=detectionModel)
detector.prepare(ctx_id=0)

# 6. Recognize faces from the frame
faceRecog = FaceIdentity(videoPath=0)


logging.info("Initiate the Camera")
# -------------------------------------------------------------

while True:

    face = faceRecog.RealTimeRecognition(getEmotions=True, getFaceLoc=True)

    logging.info("Detecting & Recognizing Faces...")
    # ---------------------------------------------------------------------------------------------

    cv2.imshow("Face Recognition", face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cam.release()
# cv2.destroyAllWindows()
