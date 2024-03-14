# ***Inmind.ai Final Project***

## **Project Overview**

## Introduction

Welcome to the inmind.ai_amazing_project's repository. 
This project is an (almost) cutting-edge solution developed to tackle the challenges of object detection within digital images. 
Leveraging state-of-the-art machine learning techniques and architectures, including YOLOv7 and custom PyTorch models, this system is designed to significantly enhance our capabilities in identifying and categorizing objects across various scenarios and datasets.

This _amazing_ project encompasses the entire pipeline of object detection tasks—from dataset preparation and augmentation to training robust models and deploying them for real-time inference. The solutions developed here demonstrate our commitment to advancing the field of computer vision, and lay a solid foundation for future innovations, to ultimately combine computer vision with <font color="red"> *robotics* </font>.

This documentation provides a comprehensive guide to the project, including setup instructions, feature highlights, usage examples, and insights into the technologies we've employed. Our goal is to offer a clear overview of the project's capabilities and facilitate its adoption and further development.

Stay tuned as we dive deeper into the details of this exciting venture into the realm of artificial intelligence and computer vision.


## Setup and Installation

Getting started with the Object Detection System project is straightforward. Follow these steps to set up your environment and run the project locally.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python (version 3.8 or higher recommended)
- Git


### Installation Steps

1. **Clone the Repository**

    First, clone the project repository to your local machine using Git:

    ```bash
    git clone https://github.com/wgtayar/inmind_amazing_project
    cd inmind_amazing_project
    ```

2. **Create a Virtual Environment (Optional but Recommended)**

    It's best practice to use a virtual environment for Python projects. This keeps your dependencies organized and avoids conflicts. To create and activate a virtual environment:

    ```bash
    python -m venv venv
    # For Windows
    venv\Scripts\activate
    # For macOS and Linux
    source venv/bin/activate
    ```

3. **Install Dependencies**

    With your virtual environment activated, install the project dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

    This command reads the `requirements.txt` file and installs all the necessary Python packages.

## Features

This object detection system is designed with the following capabilities:

- **Data Preparation and Augmentation**: Utilizes powerful libraries like Albumentations to prepare and augment images, enhancing the model's ability to generalize across different lighting conditions, angles, and backgrounds.
- **Advanced Object Detection Models**: Incorporates state-of-the-art models such as YOLOv7, alongside custom PyTorch models, ensuring high accuracy and efficiency in object detection tasks.
- **Model Training and Evaluation**: Offers a streamlined process for training object detection models, complete with evaluation metrics to assess model performance accurately.
- **Hyperparameter Optimization**: Supports experimenting with different hyperparameters to fine-tune the models for optimal performance.
- **Real-time Inference**: Capable of deploying trained models for real-time object detection, making it suitable for integration into live systems.
- **Visualization Tools**: Includes tools like TensorBoard for visualizing model metrics during training, and Netron for viewing model architectures, aiding in the interpretability and analysis of model performance.
<!-- - **Inference API**: Features a scalable API for model inference, providing endpoints for model listing, image-based detection, and returning annotated images with detected objects. -->
- **Export to Inference Models**: Enables exporting trained models to formats compatible with ONNX model, facilitating deployment across different platforms.
<!-- - **Dockerization (Optional)**: Offers the option to dockerize the inference API, simplifying deployment and scaling in production environments. -->

## Usage

This project encompasses several stages, including dataset preparation, model training, evaluation, and applying data augmentation techniques. Follow these steps to utilize the system effectively:

### Preparing Your Dataset

1. **Convert Annotations to YOLO Format**: Start by converting your dataset annotations from JSON to YOLO format, facilitating compatibility with YOLOv7 training requirements. Utilize the `convert_annotations_to_yolo_format` function provided in `ModelTraining.ipynb` for this purpose. This function reads annotations from the specified directory and converts them into YOLO format, saving the output in a designated directory.

2. **Splitting the Dataset**: To ensure the robustness of your model, split your dataset into training and validation sets. The splitting process is demonstrated in `ModelTraining.ipynb`, leveraging the `train_test_split` method from `sklearn.model_selection`.

### Training the Model

To train your object detection model, follow these steps:

1. **Loading the Dataset**: Use the `CustomObjectDetectionDataset` class from `LoadingBMWDataset.py` to load your dataset. This class allows for easy integration of custom transformations.

2. **Training**: Refer to the training process outlined in `CustomResNet.ipynb`. This notebook provides a comprehensive guide to setting up and executing the training loop with PyTorch, leveraging a custom ResNet backbone.

### Evaluating the Model

After training, evaluate your model's performance using the evaluation metrics provided in `ModelTraining.ipynb`. The `evaluate_model` function computes precision, recall, and F1 score, offering insight into your model's accuracy and reliability.

### Enhancing Dataset Robustness with Augmentations

Data augmentation is a powerful technique to improve model generalization. Use the `DatasetWithAugmentations` class in `DataAugmentation.py` to apply a series of augmentations to your dataset, as shown:

```python
dataset = DatasetWithAugmentations(img_dir, annotation_dir)
```

### Fixing Annotation Classes

If necessary, use the scripts provided in `jsonfixer.ipynb` to correct class IDs within your dataset annotations. This can be crucial for maintaining consistency and accuracy in your model's training data.

### YOLOv7 Format Conversion

For training with YOLOv7 models, ensure your annotations are in the correct format by following the conversion process outlined in `jsonfixer.ipynb`. This adaptation is essential for compatibility with YOLOv7's training requirements.

### Saving and Loading the Model

- **Saving**: Upon completing the training, save your model's state dictionary for future use:

    ```python
    torch.save(model.state_dict(), 'path_to_save_model.pth')
    ```

- **Loading**: To resume training or for evaluation, load the saved model parameters into the model architecture:

    ```python
    model.load_state_dict(torch.load('path_to_saved_model.pth'))
    ```

Follow these steps to effectively train, evaluate, and enhance your object detection models. For detailed code examples and instructions, refer to the corresponding Jupyter notebooks and Python files provided in this project.