import logging
import shutil
from src.exception import CustomException
import cv2
from src.faceDetectors.SCRFD_FaceDetector.scrfdDetector import SCRFD
from src.FaceModel.ResNet_Arch import resnet_inference
from src import config
import os
import sys
from src.FaceModel.FaceEmbeddingLoader import FaceEmbeddingLoader
import numpy as np
import torch
from torchvision import transforms
import pandas as pd

# get all the paths
detector_file = config.scrfd_weights
trained_data = config.Trained_data_dir
do_training_data = config.unTrained_data_dir

# face model path
face_model_path = config.face_model_path
backbone_model = config.face_model_backbone
feature_embeddings = config.feature_embs
recycled_folder = config.recycled_images


class FaceEmbeddingPipeline:
    def __init__(self, trained_data, do_training_data):
        self.trained_data = trained_data
        self.do_training_data = do_training_data
        self.embeddings_path = config.feature_embs

        # os.makedirs(self.embeddings_path, exist_ok=True)

        self.embedding_csv = os.path.join(self.embeddings_path, "Embeddings.csv")

        # 1. initialize detector
        self.detector = SCRFD(model_file=detector_file)
        self.detector.prepare(0)
        logging.info(f"INFO[{__name__}] SCRFD Detector initialized...")

        # face model
        self.face_model = resnet_inference(model_name=backbone_model, pretrained_model=face_model_path)
        logging.info(f"INFO[{__name__}] Face Model initialized...")

        # Check if the Cuda is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"INFO[{__name__}] Device is set to {self.device}!!")

        # get facial features from the image
        self.embedder = FaceEmbeddingLoader()

        # embeddings and names
        self.image_embs = []
        self.image_names = []

        self.image_no = 0

    def process_and_embed_faces(self):
        """
        This Function Maps the images present in the folder tree.
        Then crops the image Strored into the Trained Folder
        Then Croped Face passed to the get Embedder function that pass the facial info to ArchFace -> Returne the 512D embeddings
        These embeddings then strored into DataFrame.csv file as well as Saved as Embedder.npz

        """
        # 2. check if the data is present inside  the training is directory
        print("Checking.....")
        for name in os.listdir(self.do_training_data):
            print("In For Loop...")
            person_name_path = os.path.join(self.do_training_data, name)
            print(person_name_path)
            logging.info(f"INFO[{__name__}] Checking for images inside {person_name_path}...")
            try:
                # 2.1 if directory is present but no images Found then let the person know
                if os.path.exists(self.do_training_data) and len(os.listdir(self.do_training_data)) == 0:
                    logging.info(f"INFO[{__name__}] No Images are Found!!...")
                    print("No Images Found")
                    return None

                # 2.2 If data is present in untrained directory detect faces and create embedding
                else:
                    logging.info(f"{len(os.listdir(self.do_training_data))} Images Found !!...")
                    print(len(os.listdir(person_name_path)), "Images Found !!!")

            # 2.3 if the directory is not present then display an error
            except Exception as e:
                logging.error(f" {self.do_training_data} Does not exists... create file called '{e}' ")
                raise CustomException(e, sys)

            # 2.4 detect the face in images and get their embedding
            else:
                # Create a directory to save the faces of the person
                save_img_path = os.path.join(self.trained_data, name)
                os.makedirs(save_img_path, exist_ok=True)
                logging.info(f"INFO[{__name__}]", "Directory {0} created".format(save_img_path))

            for image_name in os.listdir(person_name_path):
                fullPath = os.path.join(person_name_path, image_name)
                if fullPath.endswith(("jpg", "png", "jpeg")):

                    # read the image
                    image = cv2.imread(fullPath)
                    logging.info(f"INFO[{__name__}]", "Reading image...")

                    # Detect faces and landmarks using the face detector
                    results = self.detector.detect(image, input_size=(640, 480))

                    if len(results) != 0:
                        boxes, landmarks = results

                        for i in range(len(boxes)):
                            x1, y1, x2, y2, _ = boxes[i]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # extract the faces
                            frame = image[y1:y2, x1:x2]
                            logging.info(f"INFO[{__name__}] Image croped...")

                            # save the extracted faces
                            save_path = f"{save_img_path}/{name}_{self.image_no}.jpg"
                            cv2.imwrite(save_path, frame)
                            logging.info(f"INFO[{__name__}] Image save at {save_path}")

                            # get the embeddings
                            # face_embeddings = get_features(frame)
                            face_embeddings = self.embedder.getEmbeddings(frame)
                            logging.info(f"INFO[{__name__}] face embeddings are ready... {face_embeddings.shape}")

                            # append those embedding to a list with respect to person's name
                            self.image_embs.append(face_embeddings)
                            self.image_names.append(name)
                            logging.info(f"INFO[{__name__}] append embeddings with respect to person's name")

                        self.image_no += 1

        # exit the loop if no images found
        if self.image_embs == [] and self.image_names == []:
            logging.info(f"INFO[{__name__}] No Data Found For Training...")
            print("No Data Found For Training...")
            return

            # convert the embedding into numpy array
        image_embs = np.array(self.image_embs)
        image_names = np.array(self.image_names)
        logging.info(f"INFO[{__name__}] converted embeddings into numpy array")

        # load the existing embeddings
        if (os.path.exists(self.embeddings_path)) and (os.listdir(self.embeddings_path) != 0) and (
                'Embeddings.npz' in os.listdir(self.embeddings_path)):
            print("Embeddings.npz loading...")
            # features = FaceEmbeddingLoader()
            load_features = self.embedder.load_embeddings()
            logging.info(f"INFO[{__name__}] Loading Saved embeddings...")
            if load_features is not None:
                prev_names, prev_embs = load_features

                # merge previous and new embeddings
                image_names = np.hstack((prev_names, image_names))
                image_embs = np.vstack((prev_embs, image_embs))
                logging.info(f"INFO[{__name__}] New images Now Merged with old images !!!")

        # save the embeddings
        os.makedirs(self.embeddings_path, exist_ok=True)
        logging.info(f"INFO[{__name__}] Creating embeddings Directory...")

        # -------------------------------------------------------------------------------------------------------------------------
        ## adding data to dataframe
        embeddings_df = pd.DataFrame(image_embs)
        logging.info(f"INFO[{__name__}] DataFrame is created.. at location {0}".format(self.embeddings_path))

        # add names in the dataframe as a new column
        embeddings_df['image_name'] = image_names
        logging.info(f"INFO[{__name__}] Names added in DataFrame... at location {0}".format(self.embeddings_path))

        # Optionally, move the 'name' column to the front
        cols = embeddings_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        embeddings_df = embeddings_df[cols]

        embeddings_df.to_csv(f"{self.embeddings_path}/Embeddings.csv", index=False)
        logging.info(f"INFO[{__name__}] DataFrame Added!!!")
        # --------------------------------------------------------------------------------------------------------------------------

        # Save the Embeddings .npz file
        embsPath = f"{self.embeddings_path}/Embeddings.npz"
        np.savez_compressed(file=embsPath, image_embs=image_embs, image_name=image_names)
        logging.info(f"INFO[{__name__}] Embeddings created at location {0}".format(self.embeddings_path))

        # move the Untrained images file to Trained Directory
        os.makedirs(recycled_folder, exist_ok=True)
        for person in os.listdir(self.do_training_data):
            dir_to_move = os.path.join(self.do_training_data, person)

            # check if the data already present in recylebin
            if dir_to_move in os.listdir(recycled_folder):
                print(os.listdir(recycled_folder))
                logging.info(f"INFO[{__name__}] Person {person} Already Present In the data...")
                pass
            else:
                shutil.move(src=dir_to_move, dst=recycled_folder, copy_function=shutil.copytree)
        logging.info(f"INFO[{__name__}] uTrained Images now Trained and moved to {0} this directory".format(recycled_folder))
        print("data has moved to {} This location".format(recycled_folder))