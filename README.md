# License Plate Detection using Convolutional Neural Networks

License Plate Detection is a fundamental task in various applications such as traffic monitoring, vehicle identification, and automatic toll collection systems. This project aims to develop a license plate detection system using Convolutional Neural Networks (CNNs) trained on a custom dataset containing images of vehicles with annotated license plate bounding boxes.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

The project focuses on building a robust license plate detection system capable of accurately localizing license plates within images. It employs deep learning techniques, specifically CNNs, to learn discriminative features of license plates and surrounding areas.

## Dataset

The dataset used for training the license plate detection model is available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection). It contains a collection of images captured from various traffic scenarios along with corresponding annotations.

1. **Images**: The dataset contains a collection of images captured from various traffic scenarios. These images are stored in the `images` directory.

2. **Annotations**: Each image in the dataset is annotated with bounding boxes around the license plates. The annotations are provided in XML format following the PASCAL VOC format. The XML files are stored in the `annotations` directory.

## Model Architecture

The license plate detection model is built using a CNN architecture. The model consists of multiple convolutional layers followed by max pooling operations to extract relevant features from the input images. The final layers include fully connected layers for classification.

The model architecture is defined in the `model.py` file.

## Training

The training process involves loading the dataset, preprocessing the images, and training the CNN model using the annotated bounding boxes as ground truth labels. The model is trained using the Adam optimizer and cross-entropy loss function.

The training process is implemented in the `trainer.py` file.

## Evaluation

After training the model, it is evaluated on a separate validation dataset to assess its performance in terms of accuracy and precision. Additionally, the model is evaluated on a test dataset to evaluate its generalization capability.

The evaluation process is implemented in the `trainer.py` file.

## Usage

To use the license plate detection system:

1. Clone this repository to your local machine.
2. Ensure you have all dependencies installed (see Dependencies section).
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) and extract it into the project directory.
4. Adjust hyperparameters and training settings in `main.py` if necessary.
5. Run `main.py` to start the training process.
6. After training, evaluate the model using the provided test dataset or your own images.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pillow


