import os
import cv2
import json
import torch
import albumentations as A
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
import torchvision.models.detection as detection
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class CustomObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]
        
        # Define a default transformation if none is provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()  # Converts image to PyTorch tensor
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # Load an image with OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        
        height, width, _ = img.shape

        annotation_file = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.json'))
        with open(annotation_file) as f:
            annotations = json.load(f)
        
        boxes = []
        labels = []

        for annotation in annotations:
            x_min = annotation['Left'] / width
            y_min = annotation['Top'] / height
            x_max = annotation['Right'] / width
            y_max = annotation['Bottom'] / height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(annotation['ObjectClassId'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        if len(boxes) == 0:
            # Log the issue and provide a default box if necessary
            print(f"No boxes found for image: {self.img_names[idx]}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        # Convert boxes to a tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Format the target dictionary

        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        return img, target