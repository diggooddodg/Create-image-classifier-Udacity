# PROGRAMMER: Ben Lepper  
# DATE CREATED: 1 May 2024                                
# REVISED DATE: 9 May 2024
# PURPOSE: Create a function that retrieves command line inputs from the user for both training and prediction.
#          The command line inputs will be received from the user using the Argparse Python module. 
#          If the user fails to provide some or all of the inputs, then the default values are
#          used for the missing inputs. 
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.
#
# Import python modules
import argparse

# Define function to retrieve input arguments for training
def get_input_args_train():
    """
    Retrieves and parses the command line arguments that may be provided by the user when
    they run the training program from a terminal window. This function uses Python's 
    argparse module to create and define the command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    Command Line Argument - positional:
      1. Data directory 
    Command line arguments - optional: 
      1. Directory to save checkpoint as --save_dir with default value 'save_directory'
      2. Model Architecture as --arch with default value 'vgg13'
      3. Use GPU for inference as --gpu with default value 'True'
      4. Learning rate as --learning_rate with default value 0.01
      5. Hidden units as --hidden_units with default value 512
      6. Epochs as --epochs with default value 5
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create positional command line argument to get data directory path
    parser.add_argument("data_directory", type=str, help="directory containing training data")

    # Create optional command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--save_dir', type = str, default = 'save_directory/', help = 'directory in which checkpint will be saved')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Model Architecture') 
    parser.add_argument('--gpu', type = bool, default = True, help = 'gpu for training - true or false')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'Hidden units') 
    parser.add_argument('--epochs', type = int, default = 5, help = 'Epochs')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

# Define function to retrieve input arguments for prediction
def get_input_args_predict():
    """
    Retrieves and parses the command line arguments that may be provided by the user when
    they run the prediction program from a terminal window. This function uses Python's 
    argparse module to create and define the command line arguments. If 
    the user fails to provide some or all of the optional arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments - positional:
      1. Path to image 
      2. Checkpoint filename 
    Command line arguments - optional: 
      1. Number of top K most likely cases as --top_k with default value of k = '3'
      2. Use GPU for inference as --gpu with default value 'True'
      3. Filename containing mapping of categories to real names as --category_names with default value 'cat_to_name.json'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create positional command line argument to get data directory path
    parser.add_argument("image_path", type=str, help="path to image")
    parser.add_argument("checkpoint_filename", type=str, help="checkpoint filename")

    # Create optional command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--top_k', type = int, default = 3, help = 'number of top possible flowers')
    parser.add_argument('--gpu', type = bool, default = True, help = 'gpu for inference - true or false')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'filename of mapping class index to flower names')
        
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()