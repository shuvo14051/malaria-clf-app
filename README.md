# Malaria Slide Image Classification

This repository contains a Convolutional Neural Network (CNN) model for binary classification of malaria slide images, as well as a Streamlit web application for deploying the model.

## Overview

Malaria is a serious infectious disease transmitted by mosquitoes, and early detection is critical for effective treatment. This project aims to automate the detection of malaria by classifying slide images as either parasitized or uninfected.

## Project Structure

- **model_training**: This directory contains the code for training the CNN model using TensorFlow and Keras.
- **streamlit_app**: This directory includes the Streamlit application code used to deploy the model.

## Dataset

The dataset used for training and testing the model is not included in this repository. You can download the dataset from [Kaggle: Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

## Model

The CNN model is built using TensorFlow and Keras. It has been trained on the malaria slide image dataset, which contains two classes:
- **Parasitized**: Images of cells infected with malaria.
- **Uninfected**: Images of healthy cells.

### Model Architecture
The model consists of multiple Conv2D layers followed by MaxPooling2D layers, and finally a Dense layer for classification. The key layers include:

- **Conv2D layers**: Extract features from the images.
- **MaxPooling2D layers**: Reduce the dimensionality of the feature maps.
- **Flatten layer**: Flatten the output from the previous layers.
- **Dense layer**: Output layer with softmax activation for classification.

## Streamlit Application

The Streamlit application provides an easy-to-use interface for users to upload an image of a cell slide and receive a classification result (Parasitized or Uninfected).

### Features

- Upload an image of a cell slide.
- The model predicts whether the cell is parasitized or uninfected.
- Displays the prediction result.

