# Udacity AI with Python project: Create-your-own-image-classifier
Purpose: To build an image classifier that will predict flower types using PyTorch. This was done for Udacity's AI Programming with Python Nanodegree.

## Part 1 - Jupyter notebook
The whole trainng and prediction workflow is contained in the Jupyter Notebook.

## Part 2 - Comand line app
There are 2 python executables for the command line app, one for training and one for prediction. These executables use modules from 5 additional files as described below. 

### Training
train.py can be executed to build and train the image classifier /n
Mandatory argument: \n
The user will need to specify one mandatory argument 'data_dir' contating the path to the training data directory as str. 
Optional arguments:
--save_dir: the saving directory.
--arch: the user can choose which architecture to use for the neural network. The default architecture is Alexnet: alternatively the user can choose to input VGG13
--learning_r: sets the Learning rate for gradient descent: default is 0.001.
--hidden_units: an int specifying how many neurons an extra hidden-layer will contain if so chosen.
--epochs: specifies the number of epochs as integer. Set to 5 by default.

### Prediction
predict.py can be executed to predict a flower type for a single image.
Mandatory arguments:
The user will need to specify the the path to the input image and the checkpoint filename.
Optional arguments:
--top_k: let's the user specify the numer of top K-classes to output. Default is 5.
--category_names: allows user to provide path of JSON file mapping categories to names.
--GPU: the user should specifify GPU if a GPU is available. The model will use the CPU otherwise.

### Additional files
train.py and predict.py use modules where necessary from the following files:
get_input_args.py contains modules to accept inputs from the user via the command line
prepare_data.py contains modules for preparing training and validation data and preparation for the image to be predicted
classifier.py contains modules to build and train the model and run a prediction
save_load_model.py contains mdolues to save and load the checkpoint
display_results.py contains modules to display the prediction results to the user
