# DATE CREATED: 2 May 2024                               
# REVISED DATE: 9 May 2024
# PURPOSE: To save models to and  load models from checkpoint files.
#          *****List functions in this file
#          Options:
#               * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#               * Choose architecture: python train.py data_dir --arch "vgg13"
#               * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#               * Use GPU for training: python train.py data_dir --gpu
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.

# import python modules
import torch
from torchvision import models
from torch import optim

# save checkpoint
def save_checkpoint(arch, model, optimizer, epochs, save_dir, class_to_idx):
    
    # define the class to output indices map
    model.class_to_idx = class_to_idx
    
    checkpoint = {'architecture': arch,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs
                  }
    
    torch.save(checkpoint, save_dir)

# load model form checkpoint and rebuild
def load_checkpoint(checkpoint_filename):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_filename)
    
    # Rebuild the model
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    
    # Load state dictionary
    model.load_state_dict(checkpoint['state_dict'])
      
    # Attach mapping of classes to indices
    model.class_to_idx = checkpoint['class_to_idx']
    
    #initialise and rebuild the optimiser
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer