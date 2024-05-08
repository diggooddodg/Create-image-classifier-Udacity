# DATE CREATED: 2 May 2024                               
# REVISED DATE: 9 May 2024
# PURPOSE: This file contains functions that will:
#           a) Create a classifier model and 
#           b) Train a classifier model
#           b) Perform predictions using the trained model
#          *****List functions in this file
#          Options:
#               * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#               * Choose architecture: python train.py data_dir --arch "vgg13"
#               * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#               * Use GPU for training: python train.py data_dir --gpu
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.

# import python modules
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import models
import time

# Import local functions
from prepare_data import process_image

# get pretrained model and add custom classifier
def create_model(arch, hidden_units):
    
    # use pretrained model per user inputs  
    # seen the following code when Googling but Udacity recommends stricter conditional structure as below
    #       exec(f"model = models.{arch}(pretrained=True)")
    #       to get in_features once model has been defined: in_features = model.classifier.in_features

    if arch == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
        in_features = 25088
    elif arch == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
        in_features = 25088
    elif arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        in_features = 25088
    elif arch ==  "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
        in_features = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)   
        in_features=9216   
   
    # Freeze parameters in the pre-trained model so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Update the classifier of the pre-trained model. 
    # Use in-features, hidden_units arguments. Specify 102 outputs and softmax output function
        
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    
    return model

# train a model
def train_model(model, learning_rate, epochs, trainloader, validationloader, gpu):
    
    # Use GPU if it has been selected and is available
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    #define the loss function
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #set Variables for training
    training_steps = 0
    training_loss = 0
    validate_every = 5 #how often validation is done i.e. every <validate_every> batches

    for epoch in range(epochs):
        for images, labels in trainloader:
            start = time.time()
            training_steps += 1
            
            #move image and label to device
            images, labels = images.to(device), labels.to(device)
            
            #perform training steps
            #optimiser gradient will accumulate so set to zero before calculating gradients each time
            optimizer.zero_grad()
            #find raw probabilities log is used here because output is an exponent (softmax)
            logps = model(images)
            #Calculate loss overall
            loss = criterion(logps, labels)
            #calculate gradients for all parameters
            loss.backward()
            #Update parameters  
            optimizer.step()
            
            training_loss += loss.item()
            
            #perform validation every <validate_every> batches
            if training_steps % validate_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                
                #disable autograd for validation - gradients do not need to be calculated
                with torch.no_grad():
                    #perform validation loop on each batch in the validation loader
                    for images, labels in validationloader:

                        #move image and label to device
                        images, labels = images.to(device), labels.to(device)

                        logps = model(images)
                        loss = criterion(logps, labels) #find the loss for this batch
                        validation_loss += loss.item() #sum losses for all batches

                        #calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim = 1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))
                              
                print(f"Device = {device} ",
                    f"Time for this batch: {(time.time() - start)/3:.3f} seconds ",
                    f" Epoch: {epoch + 1}/{epochs} ",
                    f" Batch: {training_steps}/{len(trainloader)} ",
                    f"Training Loss: {training_loss/validate_every:.3f} ",
                    f"Validation Loss: {validation_loss/len(validationloader):.3f} ", #divide by number of batches to see average loss
                    f"Validation Accuracy: {accuracy/len(validationloader):.3f}") #divide by number of batches to see average accuracy
                
                training_loss = 0
                model.train()
    return model, optimizer

# perform prediction
def predict(image_path, model, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #convert image to numpy array and normalise
    image = process_image(image_path)   
    
    # Use GPU if it has been selected and is available
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    # Convert the NumPy array to a PyTorch tensor and move to device
    image_tensor = torch.tensor(image)
    image_tensor = image_tensor.to(device)
    
    #format tensor so that it can be used by the model 
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.float()
   
    #turn off drop out
    model.eval()    

    #run the model with no gradient descent
    with torch.no_grad():
        logps = model(image_tensor)
        
    #get top classes for this image
    probs = torch.exp(logps)
    top_probs, top_indices = probs.topk(top_k, dim = 1)  
    
    #convert probabilities and indices from tensor to numpy
    np_top_probs, np_top_indices = top_probs.data.cpu().numpy()[0], top_indices.data.cpu().numpy()[0]
        
    #convert output indices to classes 
    # class_to_idx dictionary has key = class and value = index 
    # so first need to create an inverse of the class_to_idx dictionary
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    
    # use idx_to_class dictionary to find top classes from the top output indices
    top_classes = [idx_to_class[x] for x in np_top_indices] 

    return np_top_probs, top_classes
