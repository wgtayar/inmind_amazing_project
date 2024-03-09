import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import os
import json

class LoadingBMWDataset(Dataset):
    # Constructor, initializing the paths and transform 
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Getting the length of the dataset
    def __len__(self):
        return len(self.images)
    
    # Getting the sample
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.images[idx].replace('.jpg', '.json'))
        
        # Loading the image
        image = Image.open(img_path).convert("RGB")
        
        # Loading the labels
        with open(label_path) as f:
            annotations = json.load(f)
        
        # Transforming the sample to a dictionary containing the image and labels
        sample = {'image': image, 'annotations': annotations}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Function to visualize one image along with the bounding boxes and labels
def visualize_image(sample):
    image, annotations = sample['image'], sample['annotations']
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        # The bounding box coordinates
        bbox = [annotation['Left'], annotation['Top'], annotation['Right'], annotation['Bottom']]
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1]), annotation['ObjectClassName'], fill="white")
    image.show()

# Loading images and labels from the directory
images_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/images'
labels_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/labels/json'
dataset = LoadingBMWDataset(images_dir, labels_dir)

# Function showing 5 pictures
def visualize_images(dataset, num_images=5):
    for i in range(num_images):
        sample = dataset[i]  # Get the ith sample
        visualize_image(sample, i)

# Execution of the above functions to visualize the images and labels, uncomment to visualize the images with their labels
# visualize_images(dataset, num_images=5)
