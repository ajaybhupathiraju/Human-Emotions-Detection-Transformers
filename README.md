# Human-Emotions-Detection-Transformers
Human emotions detection using transformers to detect and classify emotions from images. 

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Why Transformers choosen over convolution neural networks](#why-transformers-over-cnn)
- [EDA](#exploratory-data-analysis)
- [Recommendations](#recommendations)
- [Machine learning model](#machine-learning-model)
  - [Vision Transformer build from scratch](#custom-vit-model)
  - [using pretrained hugging face model](#huggingface-tfvitmodel)
- [Assumptions](#assumptions)
- [Performance metrics](#performance-metrics)
- [Loss function](#loss-function)
- [Choosing a Model](#choosing-a-model)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Deployee model](#deployee-model)


## Project Overview
  
  The goal of this project is to build a model using vision transformers that detects human emotions (e.g., happiness, sadness, anger etc ) from images, each image is of different dimentions.

## Data Sources 
  Downloaded dataset from kaggle and its is organized into subfolders for each emotion
   - Train Folder:
      -  angry
      -  happy
      -  sad
  -  Test Folder:
      -  angry
      -  appy
      -  sad

## Why Transformers over CNN

Convolution Nueral Network                                                                 |             Transformers                                                                                                             | 
-----------------------------------------------------------------------------------------  | ----------------------------------------------------------------------------------------------------------------------------         | 
Using convolutional layers to capture local patterns in data edges & textures etc          |   Transformers don't use convolutional or pooling operations instead, they rely on fully connected layers and attention mechanisms. | 
max-pooling - reduce dimentionality of data<br>flattened   - fed data into fully connected layers for classification or regression tasks.|Attention allows models to dynamically focus on pertinent parts of the input data.                                        
smaller dataset.                                                                            |   Larger datasets.                                                                                                                 | 

