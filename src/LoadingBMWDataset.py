
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import os
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

class LoadingBMWDataset(Dataset):
    # Constructor, initialize the paths and transform 
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Get the length of the dataset
    def __len__(self):
        return len(self.images)
    
    # Get the sample
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.json'))
        
        # Load image
        image = read_image(img_path)
        
        # Load labels
        with open(label_path) as f:
            annotations = json.load(f)
        
        # Transform the sample to a dictionary containing the image and labels
        sample = {'image': image, 'annotations': annotations}
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    # Get the sample
    # def __getitem__(self, idx):
    #     img_path = os.path.join(self.images_dir, self.images[idx])
    #     label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.json'))
        
    #     # Load image
    #     image = Image.open(img_path).convert("RGB")
        
    #     # Load labels
    #     with open(label_path) as f:
    #         annotations = json.load(f)
        
    #     # Transform the sample to a dictionary containing the image and labels
    #     sample = {'image': image, 'annotations': annotations}
    #     if self.transform:
    #         sample = self.transform(sample)
        
    #     return sample

# Loading images and labels from the directory
images_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/images'
labels_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/labels/json'
dataset = LoadingBMWDataset(images_dir, labels_dir)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print('ok')
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

'''
####################################################################################################################
###COMMENTING THIS SECTION OUT TO AVOID RUNNING THE CODE WHEN IMPORTING, UNCOMMENT TO VISUALIZE IMAGES AND LABELS###
####################################################################################################################
# Function to visualize one image along with the bounding boxes and labels
def visualize_image(sample, idx):
    image, annotations = sample['image'], sample['annotations']
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        # The bounding box coordinates
        bbox = [annotation['Left'], annotation['Top'], annotation['Right'], annotation['Bottom']]
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1]), annotation['ObjectClassName'], fill="white")
    image.show()

# Function showing 5 pictures
def visualize_images(dataloader, num_images=5):
    for i in range(num_images):
        sample = dataset[i]  # Get the ith sample
        visualize_image(sample, i)

# Execution of the above functions to visualize the images and labels
visualize_images(dataloader, num_images=5)

'''