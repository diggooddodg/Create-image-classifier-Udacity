# Udacity AI with Python project: Create-your-own-image-classifier
Purpose: To build an image classifier that will predict flower types for Udacity's AI Programming with Python Nanodegree utilising pythorch.

## Part 1 - Jupyter notebook
The whole workflow is contained in this Jupyter Notebook.

## Part 2 - Comand line app
There are 2 python executables:

train.py (https://github.com/bsassoli/Create-your-own-image-classifier/blob/master/predict.py)
predict.py (https://github.com/bsassoli/Create-your-own-image-classifier/blob/master/train.py)
Training the classifier
train.py will train the classifier. The user will need to specify one mandatory argument 'data_dir' contating the path to the training data directory as str. Optional arguments:

--save_dir: the saving directory.
--arch: the user can choose which architecture to use for the neural network. The default architecture is Alexnet: alternatively the user can choose to input VGG13
--learning_r: sets the Learning rate for gradient descent: default is 0.001.
--hidden_units: an int specifying how many neurons an extra hidden-layer will contain if so chosen.
--epochs: specifies the number of epochs as integer. Set to 5 by default.
--GPU: the user should specifify GPU if a GPU is available. The model will use the CPU otherwise.
Using the classifier
predict.py will accept an image as input and will output a probability ranking of predicted flower species. The only mandatory argument is -image_dir, the path to the input image. The options are:

--load_dir: the checkpoint path.
--top_k: let's the user specify the numer of top K-classes to output. Default is 5.
--category_names: allows user to provide path of JSON file mapping categories to names.
--GPU: the user should specifify GPU if a GPU is available. The model will use the CPU otherwise.
