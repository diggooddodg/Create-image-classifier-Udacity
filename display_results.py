# PROGRAMMER: Ben Lepper  
# DATE CREATED: 3 May 2024                               
# REVISED DATE: 9 May 2024
# PURPOSE: Use a trained network to predict the flower name for given an input image
#          User will specify a single image and the flower name and class probability will be returned.
#          Basic command line usage: python predict.py /path/to/image checkpoint
#          Options:
#               * Return top K most likely classes: python predict.py input checkpoint --top_k 3
#               * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#               * Use GPU for inference: python predict.py input checkpoint --gpu
#          This file is one of several used for part 2 of the Udacity "Create your own Image classifier" project.
      
# Import python libraries
import numpy as np

# Import local functions
from prepare_data import get_mapping

# display top predicted image and graph of top_k probabilities
def display_prediction(top_classes, top_probs, category_names):  
    
    #convert classes to flower names
    cat_to_name = get_mapping(category_names)
    flower_names = [ cat_to_name[x] for x in top_classes ]

    print(f"Top prediction is: {flower_names[0]}")
    print(f"Top predictions are: {flower_names}")
    with np.printoptions(precision=5, suppress=True):
        print(f"Top prediction probabilities are{top_probs}")
           
   