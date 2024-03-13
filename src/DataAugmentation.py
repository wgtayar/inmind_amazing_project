import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


'''

class MixedObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None, augment_transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform  # For original images
        self.augment_transform = augment_transform  # For augmented images
        self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]

    def __len__(self):
        # Double the length since we are combining original and augmented images
        return len(self.img_names) * 2

    def __getitem__(self, idx):
        # Determine if we should fetch the original or augmented image
        original = idx < len(self.img_names)
        idx = idx if original else idx - len(self.img_names)

        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation_file = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.json'))
        with open(annotation_file) as f:
            annotations = json.load(f)
        
        # Prepare bounding boxes and labels
        boxes = []
        labels = []
        for annotation in annotations:
            x_min, y_min, x_max, y_max = annotation['Left'], annotation['Top'], annotation['Right'], annotation['Bottom']
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(annotation['ObjectClassId'])
        
        # Apply transformation
        if original and self.transform:
            img = self.transform(image=img)['image']
        elif self.augment_transform:
            transformed = self.augment_transform(image=img, bboxes=boxes, class_labels=labels)
            img, boxes = transformed['image'], transformed['bboxes']
        
        return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
'''

class AugmentedObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        annotation_file = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.json'))
        with open(annotation_file) as f:
            annotations = json.load(f)

        boxes = []
        for annotation in annotations:
            # Normalize the bounding box coordinates
            x_min = annotation['Left'] / width
            y_min = annotation['Top'] / height
            x_max = annotation['Right'] / width
            y_max = annotation['Bottom'] / height
            boxes.append([x_min, y_min, x_max, y_max])

        labels = [annotation['ObjectClassId'] for annotation in annotations]

        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)



# Defining the transformations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(512, 512),
    ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Running the data augmenting process through a main function
if __name__ == '__main__':
    img_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/images'
    annotations_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/labels/json'

    dataset = AugmentedObjectDetectionDataset(img_dir=img_dir, annotation_dir=annotations_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

'''
    # Printing the characteristics of image, the boxes, and the labels
    for img, boxes, labels in data_loader:
        print(img.shape, boxes, labels)
'''