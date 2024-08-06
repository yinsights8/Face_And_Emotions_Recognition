import logging
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2

class VisionTracker:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tracks = []
        self.track_id = 0

    def update(self, image, detections):
        # Prepare the image for the model
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        # Convert detections to torch tensor
        if detections.size > 0:
            detections = torch.tensor(detections[:, :4], dtype=torch.float32).to(self.device)
        else:
            detections = torch.empty((0, 4), dtype=torch.float32).to(self.device)

        # Run the detector model
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Prepare new tracks
        new_tracks = []
        for box in outputs[0]["boxes"]:
            x1, y1, x2, y2 = box.cpu().numpy()
            track = STrack(
                track_id=self.track_id,
                tlbr=[x1, y1, x2, y2],
            )
            self.track_id += 1
            new_tracks.append(track)

        self.tracks = new_tracks
        return self.tracks

class STrack:
    def __init__(self, track_id, tlbr):
        self.track_id = track_id
        self.tlbr = tlbr
        self.is_activated = True