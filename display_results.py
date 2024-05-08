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
      
# Import python  modules
import matplotlib.pyplot as plt
import numpy as np

# Import local functions
from prepare_data import process_image, get_mapping

# transform and display numpy image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension so need to transpose
    image = image.transpose((1, 2, 0))
    
    # Undo mean and std dev normalising
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# display top predicted image and graph of top_k probabilities
def display_prediction(image_path, top_classes, top_probs, category_names):  
    
    #convert classes to flower names
    cat_to_name = get_mapping(category_names)
    flower_names = [ cat_to_name[x] for x in top_classes ]
    
    #display the image
    processed_image = process_image(image_path)
    imshow(processed_image)
    plt.title(f"Top predicted flower: {flower_names[0]}")

    #display graph of top 5 probable names
       
    # define x and y values 
    x = flower_names
    y = top_probs

    # Set graph size
    plt.figure(figsize=(5, 5))

    #create horizontal bar graph
    plt.barh(x, y)

    # Label the axes
    plt.ylabel('flower')
    plt.xlabel('probability')

    #show top prediction at top
    plt.gca().invert_yaxis()

    # Dsiplay the plot
    plt.show()  