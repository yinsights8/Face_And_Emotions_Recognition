# preprocess libraries for Emotion Detection
import torch
from torchvision import transforms
import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import cv2

class Face_Preprocess:
    def __init__(self):
        # self.device = device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_Face_For_Emotions(self, FaceROI):
        """
        This function will take in a detected croped image for preprocessing
        :param croped_face:
        :param device: cuda if cuda is available else cpu
        :return: Face image tensor
        """

        # initialize a list of preprocessing steps to apply on each image during runtime
        data_transform = transforms.Compose([
            ToPILImage(),
            Grayscale(num_output_channels=1),
            Resize((48, 48)),
            ToTensor()
        ])

        face = data_transform(FaceROI)
        face = face.unsqueeze(0)
        face = face.to(self.device)

        return face

    def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
        # check if the width and height is specified
        if width is None and height is None:
            return image

        # initialize the dimension of the image and grab the
        # width and height of the image
        dimension = None
        (h, w) = image.shape[:2]

        # calculate the ratio of the height and
        # construct the new dimension
        if height is not None:
            ratio = height / float(h)
            dimension = (int(w * ratio), height)
        else:
            ratio = width / float(w)
            dimension = (width, int(h * ratio))

        # resize the image
        resized_image = cv2.resize(image, dimension, interpolation=inter)

        return resized_image
