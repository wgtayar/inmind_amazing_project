# ***Inmind.ai Final Project***

## **Project Overview**

## Introduction

Welcome to the inmind.ai_amazing_project's repository. 
This project is an (almost) cutting-edge solution developed to tackle the challenges of object detection within digital images. 
Leveraging state-of-the-art machine learning techniques and architectures, including YOLOv7 and custom PyTorch models, this system is designed to significantly enhance our capabilities in identifying and categorizing objects across various scenarios and datasets.

This _amazing_ project encompasses the entire pipeline of object detection tasks—from dataset preparation and augmentation to training robust models and deploying them for real-time inference. The solutions developed here demonstrate our commitment to advancing the field of computer vision, and lay a solid foundation for future innovations, to ultimately combine computer vision with <font color="red">*robotics*</font>.

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

## Training Metrics

Below are the TensorBoard screenshots demonstrating the training metrics and loss curves for the Yolov7 model training/

![Training Metrics](/home/wgt/Desktop/Pics to the README/Screenshot from 2024-03-14 14-26-25.jpg)

![Training Loss Curve](/home/wgt/Desktop/Pics to the README/Screenshot from 2024-03-14 14-26-35.jpg)

## ONNX Models

The ONNX models used in this project are available for download from the following OneDrive folder. These models reflect the weights acquired during training for both the Yolov7 and the customResNet models. 

[Download ONNX Models from OneDrive](https://1drv.ms/u/s!Aiat635zdKFsgotiJsqacOBhVJuJvw?e=6IFJrg)

## CustomResNet Model Architecture

The CustomResNet model extends a pre-trained ResNet50 model by integrating custom layers designed to enhance object detection capabilities. This architecture aims to leverage the robust feature extraction capabilities of ResNet50 while tailoring the model's head for specific object detection tasks.

### Architecture Overview

- **Base Model**: ResNet50, known for its deep residual learning framework, which facilitates training of deeper networks by addressing the vanishing gradient problem.
- **Custom Layers**: Sequential layers have been added to the model's head, including ReLU-activated fully connected layers, aiming at refining the feature representations for object detection.

### Inspiration and References

This approach draws inspiration from the following papers, which explore enhancements to convolutional neural network architectures for improved performance in object detection tasks:

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

These works demonstrate the effectiveness of deep residual learning and feature hierarchies in object recognition, principles that underpin the design of our CustomResNet model.


## Utilizing YAML Files with YOLOv7

YAML files play a crucial role in configuring the YOLOv7 model for training and inference. These files specify model parameters, paths to datasets, and other configuration settings that ensure the model is trained with the correct data and hyperparameters.

### Steps to Use YAML Files:

1. **Configuration**: Edit the YAML file to include the correct paths to your training and validation datasets. Additionally, set any model-specific parameters such as input size, number of classes, and hyperparameters.

2. **Training**: When initiating the training process, pass the YAML file as an argument to specify the configuration to be used. Example command:

    ```bash
    python train.py --cfg path_to_your_yaml_file.yaml
    ```

3. **Inference**: Similarly, for inference, ensure the YAML file used for training is referenced to maintain consistency in model behavior and performance.

### Best Practices:

- **Documentation**: Clearly document any changes made to the default configuration to facilitate reproducibility.
- **Version Control**: Keep versions of your YAML configurations to track modifications over time and experiment with different settings.

By carefully managing and utilizing YAML files, you can effectively control the behavior of the YOLOv7 model, optimizing it for your specific object detection tasks.

## References

This project has been informed and inspired by a variety of resources, ranging from technical guides to academic research. Below is a list of references that have contributed to the development and understanding of the technologies and methodologies used in this project:

1. Markdown Guide - Hacks: [https://www.markdownguide.org/hacks/](https://www.markdownguide.org/hacks/)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). [https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)
3. How to Train YOLOv7 on Custom Data - Paperspace Blog: [https://blog.paperspace.com/train-yolov7-custom-data/](https://blog.paperspace.com/train-yolov7-custom-data/)
4. Fine-tuning YOLOv7 on a Custom Dataset - LearnOpenCV: [https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/](https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/)
5. Understanding Git Push and 'origin' - Warp Dev: [https://www.warp.dev/terminus/understanding-git-push-origin](https://www.warp.dev/terminus/understanding-git-push-origin)
6. Online Markdown Editor - TutorialsPoint: [https://www.tutorialspoint.com/online_markdown_editor.php](https://www.tutorialspoint.com/online_markdown_editor.php)
7. How to Train a Custom Object Detection Model with YOLOv7 - Analytics Vidhya: [https://www.analyticsvidhya.com/blog/2022/08/how-to-train-a-custom-object-detection-model-with-yolov7/](https://www.analyticsvidhya.com/blog/2022/08/how-to-train-a-custom-object-detection-model-with-yolov7/)
8. Git Documentation: [https://git-scm.com/docs](https://git-scm.com/docs)
9. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

### Additional Resources on ResNet and Custom Architectures

- Implementing ResNet from Scratch: A comprehensive guide to building a Residual Network model from the ground up. (Placeholder for a real link)
- "Deep Residual Learning for Image Recognition" by Kaiming He et al. - This paper introduces the concept of deep residual learning and presents the ResNet architecture, laying the foundation for many modern deep learning approaches to computer vision. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

Please note: While direct links for creating a ResNet model from scratch are provided as placeholders, they can be replaced with actual resources or tutorials that you find most instructive and aligned with the methodologies applied in your project.


