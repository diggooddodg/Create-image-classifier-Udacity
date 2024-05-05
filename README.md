# Udacity AI with Python project: Create-your-own-image-classifier
Purpose: To build an image classifier that will predict flower types using PyTorch. This was done for Udacity's AI Programming with Python Nanodegree.

## Part 1 - Jupyter notebook
The whole trainng and prediction workflow is contained in the Jupyter Notebook.

## Part 2 - Comand line app
There are 2 python executables for the command line app, one for training and one for prediction. These executables use modules from 5 additional files as described below. 

### Training
train.py can be executed to build and train the image classifier 
##### Mandatory argument
The user will need to specify one mandatory argument 'data_dir' contating the path to the training data directory as str. 
##### Optional arguments:
--save_dir: the saving directory. Default is 'save_directory/'.  
--arch: the user can choose which architecture to use for the neural network. The default architecture is vgg13.  
--learning_r: sets the Learning rate for gradient descent: default is 0.01.  
--hidden_units: an int specifying how many neurons an extra hidden-layer will contain if so chosen. Default is 512.  
--epochs: specifies the number of epochs as integer. Set to 20 by default.  

### Prediction
predict.py can be executed to predict a flower type for a single image.  
##### Mandatory arguments:
The user will need to specify the the path to the input image and the checkpoint filename.
##### Optional arguments:
--top_k: let's the user specify the numer of top K-classes to output. Default is 3.  
--category_names: allows user to provide path of JSON file mapping categories to names. Default is cat_to_name.json.  
--GPU: Allows the user to specifify GPU = True if a GPU is available. Default is GPU = True.  

### Additional files
train.py and predict.py use modules where necessary from the following files:  
get_input_args.py contains modules to accept inputs from the user via the command line.  
prepare_data.py contains modules for preparing training and validation data and preparation for the image to be predicted.  
classifier.py contains modules to build and train the model and run a prediction.  
save_load_model.py contains mdolues to save and load the checkpoint.  
display_results.py contains modules to display the prediction results to the user.  
