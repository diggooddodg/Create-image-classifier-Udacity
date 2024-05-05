# DATE CREATED: 2 May 2024                               
# REVISED DATE: 
# PURPOSE: To prepare data used to:
#          a) train a new network and 
#          b) perform predictions on the trained network.
#          *****List functions in this file
#          Options:
#               * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#               * Choose architecture: python train.py data_dir --arch "vgg13"
#               * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#               * Use GPU for training: python train.py data_dir --gpu
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.

# Import libraries
import torch
from torchvision import datasets, transforms
import time
import numpy as np
import json
from PIL import Image 

# prepare training and validation data
def get_train_data(data_directory):
    
    # define transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_directory, transform=train_transforms)
    validation_data = datasets.ImageFolder(data_directory, transform=validation_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=102, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=102)

    return trainloader, validationloader

# get dictionary that maps categories to names
def get_mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

# Process an image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image
    image = image.resize((256, 256))
    
    # Centre crop the image
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert to a NumPy array
    np_image = np.array(image)
        
    # Normalize the image
    # normalise colour channel from 0-255 to 0-1
    np_image = np_image / 255.0
    # normalise image using mean and standard deviation of ImageNet dataset
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # transpose color channel from the last dimension (H x W x C) to the first dimension (C x H x W) for Pytorch
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
