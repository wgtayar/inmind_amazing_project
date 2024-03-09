import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None, img_names=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        if img_names is None:
            self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]
        else:
            self.img_names = img_names
        

    def __len__(self):
        return len(self.img_names)
    
    '''
    # Use this version of the function when data visualization is desired
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        annotation_file = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.json'))
        with open(annotation_file) as f:
            annotations = json.load(f)
        boxes = []
        labels = []
        class_names = []  # List to hold class names
        for annotation in annotations:
            boxes.append([annotation['Left'], annotation['Top'], annotation['Right'], annotation['Bottom']])
            labels.append(annotation['ObjectClassId'])
            class_names.append(annotation['ObjectClassName'])  # Append class name
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.transform:
            img = self.transform(img)
        return img, boxes, labels, class_names
    '''
    
    # Use this version of the function when there is a need to use a YOLOV7 model
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = read_image(img_path).float() / 255.0
        annotation_file = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.txt'))
        
        boxes = []
        labels = []
        with open(annotation_file) as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split())
                labels.append(int(class_id))
                # Convert YOLO format back to [left, top, right, bottom]
                half_width, half_height = width / 2, height / 2
                left = x_center - half_width
                top = y_center - half_height
                right = x_center + half_width
                bottom = y_center + half_height
                boxes.append([left, top, right, bottom])
                
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if self.transform:
            img = self.transform(img)
        
        return img, boxes, labels



img_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/images'
# For labels when loading the data from the JSON files
#annotations_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/labels/json'

# For labels when loading the data from the txt files
annotations_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/labels/yolo'

# transform = Compose([ToTensor()])
transform = None
train_dataset = ObjectDetectionDataset(img_dir, annotations_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def visualize_sample(img, boxes, class_names):
    plt.figure(figsize=(10, 10))
    plt.imshow(img.permute(1, 2, 0))
    ax = plt.gca()
    for box, class_name in zip(boxes, class_names):  # Use class_name for display
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], class_name, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))  # Display class name
    plt.show()


def visualize_samples(data_loader, num_samples=5):
    visualized_samples = 0
    for images, boxes, labels, class_names in data_loader:  # Now includes class_names
        for i in range(images.size(0)):
            if visualized_samples >= num_samples:
                break
            visualize_sample(images[i], boxes[i], class_names[i])  # Pass class_names[i] for visualization
            visualized_samples += 1
        if visualized_samples > num_samples:
            break


# Visualizing the data, to be used with the first version of the __getitem__ function
# visualize_samples(train_loader, 5)