# DDoS Detection using Neural Networks

## Overview
This project implements a neural network model to detect Distributed Denial of Service (DDoS) attacks using machine learning techniques on packet data. The model is designed to differentiate between benign traffic and DDoS attacks based on various features extracted from network packet data.

## Table of Contents
- [Background](#background)
- [Data](#data)
- [Training Features](#training-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Background
DDoS attacks are a significant threat to online services, making it essential to identify and mitigate such attacks effectively. This project focuses on building a model that can classify traffic as either benign or DDoS based on features derived from network packets.

## Data
The dataset used in this project is derived from the [ISCX DDoS data set](https://www.unb.ca/cic/datasets/ids-2017.html), specifically the `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file located in the `data` directory. The dataset contains labeled instances of network traffic with two classes:
- **BENIGN**: Normal traffic
- **DDoS**: Traffic that is part of a Distributed Denial of Service attack

## Training Features
The following features were selected for model training:
- Flow Duration
- Backward Packet Length Standard Deviation
- Active Mean
- Flow Inter-Arrival Time Standard Deviation
- Subflow Forward Bytes
- Total Length of Forward Packets

## Installation
To set up the environment for this project, ensure you have Python installed. Then, create a virtual environment and install the required libraries using pip:

   pip install -r requirements.txt

## Usage
To run the DDoS Detection algorithm, execute the following command in your terminal:

   python DDoSDetectionAlgorithm.py

Upon execution, the script will:
   1. Load and preprocess the data.
   2. Split the data into training and testing sets.
   3. Normalize the feature data.
   4. Define, compile, and train the neural network model.
   5. Evaluate the model's performance and print the confusion matrix and classification report.

## Results
After training the model, the evaluation metrics will be printed, including the confusion matrix and the classification report. Here are the results obtained from the model evaluation:

   Classification Report:
   
              precision    recall  f1-score   support

           0       0.96      0.99      0.97     19405
           1       0.99      0.97      0.98     25744

    accuracy                           0.98     45149
    macro avg      0.98      0.98      0.98     45149
    weighted avg   0.98      0.98      0.98     45149
