import logging
import os
import json
import torch

from src.exception import CustomException
import sys
import cv2
import numpy as np
import pandas as pd
from src import config
from torchvision import transforms
from src.FaceModel.ResNet_Arch import resnet_inference



# 1. import all necessary paths
saveEmbeddings = config.feature_embs
detectorPath = config.scrfd_weights
backbone_faceModel = config.face_model_backbone
faceRecModel = config.face_model_path
faceEmbeddings_path = config.feature_embs


class FaceEmbeddingLoader:
    def __init__(self, feature_path=saveEmbeddings, modelBackbone=backbone_faceModel,
                 faceRecModel=faceRecModel):

        # hardware configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"[{__name__}] Device {0} is ready...".format(self.device))

        # print(f" at start... {__name__} using use_cuda: {self.use_cuda}")
        # print(f"at start... {__name__} using device: {self.device}")

        # model backbone 'Resnet'
        self.modelBackbone = modelBackbone

        # faceRecognition model
        self.faceRecModel = faceRecModel

        # path to save the embeddings
        self.feature_path = feature_path
        self.embedding_path = os.path.join(self.feature_path, "Embeddings.npz")

        # face recognition model to create embeddings
        self.ArchFaceModel = resnet_inference(
            model_name=self.modelBackbone,
            pretrained_model=self.faceRecModel,
        )

        logging.info(f" [{__name__}] Face Recognition model is ready...")

    @torch.no_grad()
    def getEmbeddings(self, face_image):
        """
            Extract facial features from an image using the face recognition model.

            Args:
                face_image (numpy.ndarray): Input facial image.

            Returns:
                numpy.ndarray: Extracted facial features.
            """

        # image preprocessing steps
        preprocess_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        logging.info(f" [{__name__}] Creating face features")

        # convert it into RGB format
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        logging.info(f" [{__name__}] converting image back to BGR to RGB")

        # preprocess the image
        precessed_image = preprocess_image(face_image).unsqueeze(0).to(self.device)
        logging.info(f" [{__name__}] Preprocess the image")

        # get the facial features of 512 D by model
        face_embeddings = self.ArchFaceModel(precessed_image)[0].cpu().numpy()
        logging.info(f" [{__name__}] facial Features are ready of size {face_embeddings.shape}...")

        # normalize the featues
        norm_face_embeddings = face_embeddings / np.linalg.norm(face_embeddings)
        logging.info(f" [{__name__}] Features are now normalized..")

        return norm_face_embeddings

    def load_embeddings(self):
        try:
            # ----------------------------------------------------**********--------------------------------------------------------
            if not os.path.exists(self.embedding_path):
                logging.info(f"[{__name__}] Embeddings file not found at {self.embedding_path}. Generating new embeddings...")
                # Generate empty embeddings file
                np.savez(self.embedding_path, image_name=np.array([]), image_embs=np.array([]))
            # -----------------------------------------------------**********--------------------------------------------------------
            
            embs = np.load(self.embedding_path, allow_pickle=True)
            image_names = embs['image_name']
            image_embs = embs["image_embs"]
            logging.info(f"[{__name__}] Embeddings are loaded..")

            # ---------------------------------------------------- Save Embeddings in DataFrame ----------------------------------------------------

            # convert embeddings into dataframe
            embeddings_df = pd.DataFrame(image_embs)
            logging.info(f"[{__name__}] DataFrame is created.. at location {0}".format(self.feature_path))

            # add names in the dataframe as a new column
            embeddings_df['image_name'] = image_names
            logging.info(f"[{__name__}] Names added in DataFrame... at location {0}".format(self.feature_path))

            # Optionally, move the 'name' column to the front
            cols = embeddings_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            embeddings_df = embeddings_df[cols]

            embeddings_df.to_csv(f"{self.feature_path}/Embeddings.csv", index=False)
            logging.info(f"[{__name__}] DataFrame of Embeddings are created and saved at {0}".format(self.feature_path))

            # ---------------------------------------------------- Save Embeddings in DataFrame ----------------------------------------------------


            logging.info(f"[{__name__}] Embeddings loaded from {0}".format(self.feature_path))
            logging.info(f"[{__name__}] Two Files are present in {0} : {1} & {2}".format(self.feature_path, embs.files[0], embs.files[1]))
            return image_names, image_embs

        except Exception as e:
            raise CustomException(e, sys)

    def CompairEmbeddings(self, camFaceEmbeddings, StoredFaceEmbeddings):

        """
        This calculates the dot product between the given embeddings and each embedding in the embeddings array.

        :param camFaceEmbeddings: image coming from camera
        :param StoredFaceEmbeddings: image encoding which are already present in the database
        :return:Tuple of  MostSimilarEmbedding SCORE, Higher_Cosine_Similarity INDEX
        """
        # The dot product between two normalized vectors is equivalent to their cosine similarity.
        # Since the embeddings are typically normalized (i.e., they have unit length),
        # this dot product gives a measure of similarity.
        CosineSimilarity = np.dot(StoredFaceEmbeddings, camFaceEmbeddings.T)
        logging.info(f"[{__name__}] Cosine similarity Calculated...")

        # Finding the Most Similar embeddings
        # Higher values indicate greater similarity.
        Higher_Cosine_Similarity_index = np.argmax(CosineSimilarity)
        logging.info(f"[{__name__}] Found the Most Similar embeddings...")

        # retrive similarity score
        # The highest similarity score is extracted, which indicates how similar
        # the input embeddings is to the most similar precomputed embeddings
        MostSimilarEmbeddig_Score = CosineSimilarity[Higher_Cosine_Similarity_index]
        logging.info(f" [{__name__}] Got Most Similar Embeddings score and their index")

        # The function returns the highest similarity score and the index of the most similar precomputed Embedding.
        return MostSimilarEmbeddig_Score, Higher_Cosine_Similarity_index
