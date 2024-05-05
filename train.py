# PROGRAMMER: Ben Lepper  
# DATE CREATED: 1 May 2024                               
# REVISED DATE: 
# PURPOSE: To train a new network on a dataset and save the model as a checkpoint.
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.

# Import local functions
from get_input_args import get_input_args_train
from prepare_data import get_train_data, get_mapping
from classifier import create_model, train_model
from save_load_model import save_checkpoint

# Main program function defined below
def main():
    # Get input arguments for training
    train_arg = get_input_args_train()
        
    # Prepare training and validation data
    trainloader, validationloader = get_train_data(train_arg.data_directory)
    cat_to_name = get_mapping(train_arg.category_names)

    # Initialise and train model
    model = create_model(train_arg.arch)
    model, optimizer = train_model(model, train_arg.learning_rate, train_arg.epochs, trainloader, validationloader)

    # Save checkpoint
    save_checkpoint(train_arg.arch, model, cat_to_name, optimizer, train_arg.epochs, train_arg.save_dir)

# Call to main function to run the program
if __name__ == "__main__":
    main()