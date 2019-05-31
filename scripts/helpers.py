#############
# Libraries #
#############

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
torch.manual_seed(7)
from model import CycleGenerator
import PIL
import cv2

MODEL_PATH = '../model/CGAN_checkpoint.pth'
IMAGE_INPUT_PATH = '../images/inputs/'
IMAGES_OUTPUT_PATH = '../images/outputs/'

# Check GPU avalaibility
gpu_available = torch.cuda.is_available()

def load_checkpoint():
    """
    Load a checkpoint file.
    
    returns
    -------
    A checkpoint file.
    """
    # check if gpu is available
    if gpu_available:
        CGAN_checkpoint = torch.load(MODEL_PATH) # load to GPU
        print('Using GPU')
    else:
        CGAN_checkpoint = torch.load(MODEL_PATH, map_location = 'cpu') # load to CPU
        print('Using CPU')
    return CGAN_checkpoint

def load_generator(g_conv_dims, n_res_blocks, load_g):
    """
    Create and load a cycle generator.
    
    params
    ------
    g_conv_dims -- Number of features.
    n_res_blocks -- Number of residual blocks.
    load_g -- Indicate which model to load (X->y | Y->x).
    
    returns
    -------
    A CycleGAN model.
    """
    # instanciate model
    CycleGAN = CycleGenerator(g_conv_dims, n_res_blocks)
    # load checkpoint
    CGAN_checkpoint = load_checkpoint()
    # load model
    if load_g == 'G_XtoY':
        CycleGAN.load_state_dict(CGAN_checkpoint['G_XtoY_state_dict'])
    elif load_g == 'G_YtoX':
        CycleGAN.load_state_dict(CGAN_checkpoint['G_YtoX_state_dict'])
    else:
        raise NotImplementedError('Unknow {} model'.format(load_g))
    print('Generator loaded')
    return CycleGAN

def scale(x, feature_range=(-1, 1)):
    """
    Scales the input image x into the range (-1, 1).
    
    params
    ------
    x -- Input image (must be already scaled from 0-255)
    feature_range -- Default range to scale the image.
    
    returns
    -------
    A scaled image x.
    """
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x

def process_image(image_name):
    """
    Pre-process an input image.
    
    params
    ------
    image_name -- Name of the input image.
    
    returns
    -------
    A pre-process image for the model.
    """
    # open 64x64 image
    image = Image.open(IMAGE_INPUT_PATH + image_name)
    # convert to RGB
    image = image.convert('RGB')
    # resize image
    image = transforms.functional.resize(image, (128, 128))
    # convert to tensor
    image = transforms.functional.to_tensor(image).unsqueeze(0)
    # scale
    image = scale(image)
    return image

def convert_image(image):
    """
    Convert a tensor image into a numpy representation.
    
    params
    ------
    image -- Tensor image.
    
    returns
    -------
    A numpy image.
    """
    # detach and convert to numpy
    image = image.detach().cpu().squeeze(0).numpy()
    # transpose dimensions
    image = np.transpose(image, (1, 2, 0))
    # convert into image
    image = ((image + 1)*255 / (2)).astype(np.uint8)
    return image

def show_results(original, result, filename):
    """
    Plot the convert image, alongside the original. Then the results (comparison)
    and new image are both saved.
    
    params
    ------
    original -- Original tensor image.
    result -- Converted tensor image.
    filename -- Name to use for save the results.
    
    returns
    -------
    Two saved images.
    """
    # convert the tensor images into numpy
    original = convert_image(original)
    result = convert_image(result)
    # set plot parameters
    f, arr = plt.subplots(1, 2, figsize = (15, 5))
    f.tight_layout()
    arr[0].imshow(original, aspect = 'auto')
    arr[0].set_title('Original Image')
    arr[1].imshow(result, aspect = 'auto')
    arr[1].set_title('Result Image')
    plt.show()
    
    # save plot and result image to outputs
    print('Saving results')
    f.savefig(IMAGES_OUTPUT_PATH + filename + '_comparison.svg', format = 'svg', dpi = 300, bbox_inches = 'tight')
    plt.imshow(result, aspect = 'auto')
    plt.axis('off')
    plt.savefig(IMAGES_OUTPUT_PATH + filename + '_result.svg', format = 'svg', dpi = 300, bbox_inches = 'tight')
    print('Done')