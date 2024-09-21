# Human-Emotions-Detection-Transformers
Human Emotions Detection using Transformers detect and classify emotions from images

## Table of Contents
- [Project Overview](#project-overview)
- [Why Transformers choosen over convolution neural networks](#transformers-cnn)
- [Data Sources](#data-sources)
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
  
  The goal of this project is to build a model that detects human emotions (e.g., happiness, sadness, anger etc ) from images.
  downloaded dataset from kaggle contains train and test.
  
  Kaggle dataset is organized into subfolders for each emotion
   - Train Folder:
      -  angry
      -  happy
      -  sad
  -  Test Folder:
      -  angry
      -  appy
      -  sad
  
