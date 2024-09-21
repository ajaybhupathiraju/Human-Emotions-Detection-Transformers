# Human-Emotions-Detection-Transformers
Human emotions detection using transformers to detect and classify emotions from images. 

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Why Transformers choosen over convolution neural networks](#why-transformers-over-cnn)
- [EDA](#exploratory-data-analysis)
- [Machine learning model](#machine-learning-model)
  - [Vision Transformer from scratch](#custom-vit-model)
  - [Using pretrained hugging face model](#hugging-face-pretrained-model)
- [Assumptions](#assumptions)
- [Performance metrics](#performance-metrics)
- [Loss function](#loss-function)
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

## Exploratory Data Analysis

![alt text](images/EDA_HumanEmotions.jpg)

## Machine learning model
   ## custom vit model
   - [**Code for Vision Transformer scratch **](VITModel.ipynb)
     
   ## hugging face pretrained model
   - [**Using Hugging Face pre trained model **](VITModel.ipynb)
    

## Assumptions

## Performance metrics

## Loss function

## Making predictions

## Deployee model
Deployee the model into my local machine using fastAPI.

[**source code**](/deployement/ml_api.py)

Running a fastAPI server:

![alt text](images/deployee_to_fastAPI.jpg)


Test Model:
input : [40,1,2,140,289,0,0,172,0,0.0,1]

![alt text](images/test_ml_model.jpg)


Results:

![alt text](images/results.jpg)
