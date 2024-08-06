# face detector Weights
import os
import pandas as pd

scrfd_weights = "src/faceDetectors/SCRFD_FaceDetector/Weights/scrfd_2.5g_bnkps.onnx"
yolo_face_weights = "src/faceDetectors/yoloFaceDetector/yolov8m-face.pt"

Trained_data_dir = os.path.join("dataset", "Trained_data")
unTrained_data_dir = os.path.join("dataset", "unTrained_data")
feature_embs = os.path.join("dataset", "Embeddings")
# os.makedirs(feature_embs, exist_ok= True)

recycled_images = "dataset/RecycleBin"
face_model_path = "src/FaceModel/ArchWeights/arcface_r100.pth"
face_model_backbone = "r100"

no_of_faces = 10

# Emotion Detection Paramers
emotion_model_path = "src/Analytics/Emotions/Trained_Model/model.pth"
