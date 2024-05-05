# PROGRAMMER: Ben Lepper  
# DATE CREATED: 1 May 2024                               
# REVISED DATE: 
# PURPOSE: Use a trained network to predict the flower name for given an input image
#          User will specify a single image and the flower name and class probability will be returned.
#          Basic command line usage: python predict.py /path/to/image checkpoint
#          Options:
#               * Return top K most likely classes: python predict.py input checkpoint --top_k 3
#               * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#               * Use GPU for inference: python predict.py input checkpoint --gpu
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.        

# Import local functions
from get_input_args import get_input_args_predict
from prepare_data import process_image, get_mapping
from save_load_model import load_checkpoint
from classifier import predict
from display_results import format_prediction_results, display_prediction

def main():

    # Get input arguments for training
    predict_arg = get_input_args_predict()
   
    # Load and rebuild model from checkpoint
    model, optimizer = load_checkpoint(predict_arg.checkpoint_filename)
    cat_to_name = get_mapping(predict_arg.category_names)

    # Run prediction and format results
    top_probs, top_classes = predict(predict_arg.image_path, model, predict_arg.top_k, predict_arg.gpu)
    flower_names, top_probs = format_prediction_results(top_probs, top_classes, cat_to_name)

    # Display results
    display_prediction(predict_arg.image_path, flower_names, top_probs, top_classes, cat_to_name)

# Call to main function to run the program
if __name__ == "__main__":
    main()