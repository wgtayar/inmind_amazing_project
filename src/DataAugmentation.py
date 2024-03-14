import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class DatasetWithAugmentations(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg') or img_name.endswith('.png')]
        
        # Define the default transformation if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        annotation_file = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.json').replace('.png', '.json'))
        with open(annotation_file) as f:
            annotations = json.load(f)

        boxes = [[anno['Left'], anno['Top'], anno['Right'], anno['Bottom']] for anno in annotations]
        labels = [anno['ObjectClassId'] for anno in annotations]
        
        # Apply transformations
        transformed = self.transform(image=img, bboxes=boxes, labels=labels)
        img = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['labels']
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return img, target

# Example instantiation of the dataset class
img_dir = 'path_to_your_images'
annotation_dir = 'path_to_your_annotations'
dataset = DatasetWithAugmentations(img_dir, annotation_dir)

#####################################################################################################################
#                       When using this dataset with a DataLoader, uncomment the collate_fn:                        #
#####################################################################################################################
# from torch.utils.data.dataloader import default_collate                                                           #        
#                                                                                                                   #
# def collate_fn(batch):                                                                                            #
#     images = [item[0] for item in batch]  # Extract images                                                        #
#     targets = [item[1] for item in batch]  # Extract targets                                                      #
#                                                                                                                   #    
#     images = torch.stack(images, dim=0)  # Stack images to create a batch                                         #
#                                                                                                                   #
#     # Targets do not need stacking since they are already in a list of dictionaries.                              #
#     # If you had any tensor operations to perform on targets, you would do them here.                             #
#                                                                                                                   #
#     return images, targets                                                                                        #
#                                                                                                                   #
#####################################################################################################################