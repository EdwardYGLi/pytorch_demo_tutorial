# Pytorch Demo

This repo is intended to be a demo repo for training a simple pytorch model using a simple data set with hydra configuration management and visualize using weights and biases. 

## Setup

Dependencies are managed with poetry. Install poetry with : `pip install poetry` within your virtual environment. 

Install dependencies with :

`poetry install` 

## Training script

The entry point for training is in `train.py`. To change configurations, go to the `config/` directory and change any configurations there. 

## Dataset. 

The dataset here is stored in DVC within my google drive account. To download the original data, visit (https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda). Here I took the last 100 images from each directory as validation and first 900 as training. 

