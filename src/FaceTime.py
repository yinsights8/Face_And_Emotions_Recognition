import logging
import numpy as np
from src.FaceModel.ResNet_Arch import resnet_inference
from src.faceDetectors.SCRFD_FaceDetector.scrfdDetector import SCRFD
from src.FaceModel.FaceEmbeddingLoader import FaceEmbeddingLoader
from src.face_align import norm_crop
import torch
from src.tracker.sort import Sort
from src.Analytics.Emotions.EmotionNet import EmotionModel
import cvzone as cvz
import cv2
import time
from src import config
import torch.nn.functional as nnf
from src.Analytics.Emotions.preprocess import Face_Preprocess

# 1. import all necessary paths
detectorPath = config.scrfd_weights
backbone_faceModel = config.face_model_backbone
faceRecModel = config.face_model_path
faceEmbeddings_path = config.feature_embs

# Emotion Detection
emotion_model_path = config.emotion_model_path


# 2. preprare recongnition software
class FaceIdentity:
    def __init__(
            self, videoPath, detectionModel=detectorPath, cxt_id=0,
            modelBackbone=backbone_faceModel,
            faceRecModel=faceRecModel, faceEmbs_path=faceEmbeddings_path):

        self.videoPath = videoPath

        # # initialize camera from cv2
        self.cam = cv2.VideoCapture(self.videoPath)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
        self.TrackID = 0

        # set cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logging.info(f"INFO[{__name__}] Device {0} is ready...".format(self.device))

        # set face detector
        self.detector = SCRFD(model_file=self.detectionModel)
        self.detector.prepare(ctx_id=self.cxt_id)
        logging.info(f"INFO[{__name__}] Face detection model is ready...")

        # face recognition model to create embeddings
        self.FaceRecognition = resnet_inference(
            model_name=self.modelBackbone,
            pretrained_model=self.ArchFaceModel
        )
        logging.info(f"INFO[{__name__}] Face Recognition model is ready...")

        # load Face embeddings
        self.featureEmbs = FaceEmbeddingLoader()
        self.pre_image_name, self.pre_image_embs = self.featureEmbs.load_embeddings()
        logging.info(f"INFO[{__name__}] Embeddings loadded successfully...")

        # deepsort tracker
        self.tracker = Sort(max_age=20, iou_threshold=0.3)
        logging.info(f"INFO[{__name__}] Tracker Created ...")

        # empty canvas to plot emotion recognition
        self.canvas = np.zeros((300, 300, 3), dtype="uint8")
        logging.info(f"INFO[{__name__}] Blank Canvas Created...")

    # video will automatically turn off after closing website
    def __del__(self):
        return self.cam.release()

    def GetSimilariyScoreAndIndex(self, AlignedFaces):
        """
         Convert the detected face into embeddings.

        Args:
            AlignedFaces: an AlignedFaces input face image from Camera.

        Returns:
            tuple: A tuple containing the recognition score and person identity.
        """

        # get the embeddings of a person from Front camera
        AlignedFaces = self.featureEmbs.getEmbeddings(AlignedFaces)
        logging.info(f"INFO[{__name__}] collecting front embeddings from camera...")

        # compair the embeddings that are already present in database
        score, index = self.featureEmbs.CompairEmbeddings(AlignedFaces, self.pre_image_embs)
        identity = self.pre_image_name[index]
        logging.info(f"INFO[{__name__}] Collected and calculated Cosine similarity...")
        return score, identity

    def RealTimeRecognition(self, getEmotions=False, getFaceLoc=False):
        """
        Detect and Identify the Person from Camera

        :param Frame: Input Frames from Real Time Camera
        :param FaceFromCamera: input Frames
        :param getEmotions: To get the emotion bars
        :param isGetIdentity: Get the Name and trackid of the person
        :return: numpy.ndarray: The frame with recognized faces and their names annotated.
        """


        success, FaceFromCamera = self.cam.read()
        if not success:
            logging.info(f"[{__name__}] Camera could not started...")
            return "Failed to Open Camera"
        logging.info(f"INFO[{__name__}] Frames has been readed")

        FaceROI = FaceFromCamera.copy()
        detections = np.empty((0, 5))
        logging.info(f"Empty detections are created to update tracker...")
        results = self.detector.detect(FaceFromCamera, input_size=(640, 480))

        # print(f"INFO[{__name__}] Detection results: {results}")
        logging.info(f"INFO[{__name__}] Detection resutls..")

        # Initialize warning flag
        multiple_faces_warning = False

        if len(results) != 0:
            boxes, landmarks = results

            # Check if more than one face is detected
            if len(boxes) > 1:
                multiple_faces_warning = True
        
        # ------------------------------------------------Detection Tracker------------------------------------------------
            # Prepare the detections for the tracker
            for i in range(len(boxes)):
                x, y, w, h, cnf = boxes[i]
                detections = np.vstack((detections, [x, y, w, h, cnf]))

            TrackerResults = self.tracker.update(detections)
            logging.info(f"INFO[{__name__}] Tracker Updated")


            # Process each tracked face
            for result in TrackerResults:
                x1, y1, w, h, trackID = result
                x2, y2 = w, h
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                face_bbox = [x1, y1, x2, y2]
                
                if face_bbox is not None:
                    # Align the face
                    index = np.argmin([((x1 - x) ** 2 + (y1 - y) ** 2) for (x, y, _, _, _) in boxes])
                    aligned_face = norm_crop(img=FaceFromCamera, landmark=landmarks[index])
                    logging.info(f"INFO[{__name__}] Aligned the Faces")

                    # Recognize the face
                    cosineScore, Name = self.GetSimilariyScoreAndIndex(AlignedFaces=aligned_face)
                    logging.info(f"INFO[{__name__}] Calculated the Cosine Score and Got the Name")

                    if Name is not None:
                        if cosineScore < 0.25:
                            caption = "UnKnown"
                            CosineScore = f"{cosineScore: .2f}"
                            NameColor = (0, 29, 255)  # Red light color
                            RectColor = (0, 29, 255)  # Red light color
                            WarningColor = (0, 0, 255) # Red color
                            logging.info(f"INFO[{__name__}] Unknown Entity Detected")
                        else:
                            caption = f"{Name}"
                            CosineScore = f"{cosineScore: .2f}"
                            NameColor = (0, 255, 0)  # green light color
                            RectColor = (0, 255, 0)  # green light color
                            WarningColor = (0, 0, 255) # Red color
                            logging.info(f"INFO[{__name__}] Known Entity Detected")

                        if getFaceLoc:
                            cvz.cornerRect(img=FaceFromCamera, bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), l=20, t=3, colorR=RectColor)

                            text = f"{int(trackID)}: {caption}"
                            cvz.putTextRect(FaceFromCamera, text=text, pos=(max(35, int(x1)), max(35, int(y1) - 15)), scale=0.8,
                                            thickness=2, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorT=NameColor, colorR=None, colorB=None)

                        if getEmotions:
                            # Correcting the coordinates to get the region of interest
                            FROI = FaceROI[y1:y2, x1:x2]
                            logging.info(f"INFO[{__name__}] Cropped the Frame to get the region of interest")

                            emotion_prob, emotion_value = self.EmotionDetection(Face_roi=FROI, draw=True)
                            logging.info(f"INFO[{__name__}] EmotionDetection Started...")

                            FaceFromCamera = drawProbEmotion(original_frame=FaceFromCamera,
                                                            StudentName=caption, trackID=trackID,
                                                            emotion_prob=emotion_prob,
                                                            emotion_value=emotion_value)
                            logging.info(f"INFO[{__name__}] Draw Emotions on canvas...")
                # ------------------------------------------------Detection Tracker------------------------------------------------

                # Calculte FPS
                self.cTime = time.time()
                fps = 1 / (self.cTime - self.pTime)
                self.pTime = self.cTime
                logging.info(f"INFO[{__name__}] Getting FPS")

                cvz.putTextRect(FaceFromCamera, f"FPS: {int(fps)}", (50, 50), 1, 2, colorT=NameColor)

            # Draw warning if multiple faces are detected
            if multiple_faces_warning:
                cv2.putText(FaceFromCamera, "WARNING: Multiple faces detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                WarningColor, 2, cv2.LINE_AA)


        ret, buffer = cv2.imencode(".jpg", FaceFromCamera)
        logging.info(f"INFO[{__name__}] Encodding the frames...")
        FaceFromCamera = buffer.tobytes()  

        return FaceFromCamera

    # -------------------------------------- Emotion Detection Code -------------------------------------------------------------------

    def EmotionDetection(self, Face_roi, draw=False):
        # dictionary mapping for different outputs
        emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral",
                        4: "Sad", 5: "Surprised"}

        model = EmotionModel(model_path=emotion_model_path, num_of_classes=len(emotion_dict), num_of_channels=1)
        logging.info(f"INFO[{__name__}] Emotion model Initialized...")

        preprocessF = Face_Preprocess()
        logging.info(f"INFO[{__name__}] Preprocesed the Emotion Model")

        # infer the face (roi) into our pretrained model and compute the
        # probability score and class for each face and grab the readable
        # emotion detection
        face = preprocessF.preprocess_Face_For_Emotions(FaceROI=Face_roi)

        # infer the face (roi) into our pretrained model and compute the
        # probability score and class for each face and grab the readable
        # emotion detection
        predictions = model(face)
        prob = nnf.softmax(predictions, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        top_p, top_class = top_p.item(), top_class.item()

        # grab the list of predictions along with their associated labels
        emotion_prob = [p.item() for p in prob[0]]
        emotion_value = emotion_dict.values()
        # print(self.emotion_value, self.emotion_prob)

        return emotion_prob, emotion_value


def drawProbEmotion(original_frame, StudentName, trackID, emotion_prob, emotion_value):
    paddingBtext = 20

    # Transparency level
    alpha = 0.5

    # Create a blank canvas
    blank_canvas = np.zeros((300, 300, 3), dtype="uint8")

    # Size and position of the overlay
    overlay_size = (300, 180)
    overlay_position = (float(original_frame.shape[1]) - overlay_size[0] - 5, 5)  # top right corner
    # overlay_position = (cam.get(cv2.CAP_PROP_FRAME_WIDTH) - overlay_size[0] - 5, 5)  # top right corner

    # Resize the canvas to fit the desired overlay size
    resized_canvas = cv2.resize(blank_canvas, overlay_size)

    # Draw on the resized canvas
    for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
        prob_text = f"{emotion}"
        prob_per = f"{prob * 100:.2f}%"

        full_width = 150
        TopOffSet = 55
        width = int(prob * full_width)

        # Define the position for the text
        text_x = 2
        text_y = (i * paddingBtext) + TopOffSet

        # Define the position for the rectangle relative to the text
        rect_x1 = text_x + 80  # Adjust this value to position the rectangle in front of the text
        rect_x2 = rect_x1 + width  # The width of the rectangle is based on the probability

        # Fixed width of the border rectangle (100%)
        border_rect_x2 = rect_x1 + full_width
        heighOfBar = 4
        ofsetFromTop_for_text = 4

        # change the color if the probability is high
        if int(prob * 100) > int(65):
            TextColor = (0, 255, 0)
            fillRect = (0, 255, 0)
        else:
            TextColor = (255, 255, 255)
            fillRect = (0, 255, 255)

        # Draw the filled yellow rectangle with reduced height and width
        cv2.rectangle(img=resized_canvas, pt1=(rect_x1, text_y - heighOfBar), pt2=(rect_x2, text_y + 5),
                      color=fillRect, thickness=cv2.FILLED)

        # Draw the white border rectangle with fixed width (100%)
        cv2.rectangle(img=resized_canvas, pt1=(rect_x1, text_y - heighOfBar), pt2=(border_rect_x2, text_y + 4),
                      color=(255, 255, 255), thickness=1)

        cv2.putText(resized_canvas, prob_text, (text_x, text_y + ofsetFromTop_for_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 1, cv2.LINE_AA)

        cv2.putText(resized_canvas, prob_per, (border_rect_x2 + 10, text_y + ofsetFromTop_for_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fillRect, 1, cv2.LINE_AA)

    # captions
    caption = StudentName.replace("_", " ")
    caption = f"Student Name: {caption.upper()}"
    TrackId = f"TrackID: {trackID}"

    cv2.putText(resized_canvas, caption, (text_x, 15 + ofsetFromTop_for_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 1, cv2.LINE_AA)

    cv2.putText(resized_canvas, TrackId, (text_x, 35 + ofsetFromTop_for_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 1, cv2.LINE_AA)

    # Get the region of interest (ROI) in the frame where we want to overlay the canvas
    y1, y2 = int(overlay_position[1]), int(overlay_position[1] + overlay_size[1])
    x1, x2 = int(overlay_position[0]), int(overlay_position[0] + overlay_size[0])
    roi = original_frame[y1:y2, x1:x2]

    # Blend the resized canvas with the ROI
    blended = cv2.addWeighted(roi, 1 - alpha, resized_canvas, alpha, 0)

    # Place the blended region back into the frame
    original_frame[y1:y2, x1:x2] = blended

    return original_frame

# def drawProbEmotion(emotion_prob, emotion_value):
#     # Emotion canvas
#     canvas = np.zeros((300, 300, 3), dtype="uint8")

#     # draw the probability distribution on an empty canvas initialized
#     for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
#         prob_text = f"{emotion}: {prob * 100:.2f}%"
#         width = int(prob * 300)
#         cv2.rectangle(canvas, (5, (i * 50) + 5), (width, (i * 50) + 50),
#                       (0, 0, 255), -1)
#         cv2.putText(canvas, prob_text, (5, (i * 50) + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#     return canvas
