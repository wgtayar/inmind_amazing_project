from LoadingBMWDataset import LoadingBMWDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import json

# Defining the transformations
transform = A.Compose([
    A.HorizontalFlip(p=0.5), # Flip the image horizontally
    A.RandomBrightnessContrast(p=0.2), # Randomly change the brightness and contrast of the image
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9), # Randomly shift, scale, and rotate the image
    ToTensorV2() # Convert the image to a PyTorch tensor
])

# Loading images and labels from the directory
images_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/images'
labels_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/Training/labels/json'

# Instantiating the LoadingBMWDataset with the transformations
dataset = LoadingBMWDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)

# Using the dataset to visualize the augmented images and labels
for i in range(5):
    sample = dataset[i]
    print(sample['image'].shape, sample['annotations'])

'''
##################################################################################################
### COMMENTING THIS SECTION OUT. UNCOMMENT IF THE AUGMENTED IMAGES AND LABELS NEED TO BE SAVED ###
##################################################################################################

# Function to save the augmented images and labels
def save_augmented_data(dataset, save_dir, num_images=5):
    images_save_dir = os.path.join(save_dir, 'images')
    labels_save_dir = os.path.join(save_dir, 'labels', 'json')
    
    # Creating the directories to save the images and labels
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)

    # Looping through the dataset to save the images and labels
    for i in range(num_images):
        sample = dataset[i]
        image, annotations = sample['image'], sample['annotations']
        
        # Converting the PyTorch tensor image back to an np array and then to an image
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype('uint8')
        
        # Saving the image using OpenCV
        cv2.imwrite(f"{images_save_dir}/{i}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save the labels in JSON format
        label_filename = f"{labels_save_dir}/{i}.json"
        with open(label_filename, 'w') as label_file:
            json.dump(annotations, label_file, indent=4)

# Saving the augmented images and labels to a folder called ExtraTraining
save_dir = '/home/wgt/Desktop/InMind Academy/AI_Track/Amazing_Project/inmind_amazing_project/data/ExtraTraining'
save_augmented_data(dataset, save_dir, num_images=5)
'''