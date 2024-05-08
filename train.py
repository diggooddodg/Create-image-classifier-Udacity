# PROGRAMMER: Ben Lepper  
# DATE CREATED: 1 May 2024                               
# REVISED DATE: 9 May 2024
# PURPOSE: To train a new network on a dataset and save the model as a checkpoint.
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.

# Import local functions
from get_input_args import get_input_args_train
from prepare_data import get_train_data
from classifier import create_model, train_model
from save_load_model import save_checkpoint

# Main program function defined below
def main():
    # Get input arguments for training
    train_arg = get_input_args_train()
        
    # Prepare training and validation data
    trainloader, validationloader, class_to_idx = get_train_data(train_arg.data_directory)
    
    # Initialise and train model
    model = create_model(train_arg.arch, train_arg.hidden_units)
    model, optimizer = train_model(model, train_arg.learning_rate, train_arg.epochs, trainloader, validationloader, train_arg.gpu)

    # Save checkpoint
    save_checkpoint(train_arg.arch, model, optimizer, train_arg.epochs, train_arg.save_dir, class_to_idx)

# Call to main function to run the program
if __name__ == "__main__":
    main()