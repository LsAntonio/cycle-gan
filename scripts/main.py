#############
# Libraries #
#############

from helpers import process_image, load_generator, show_results
import numpy as np
import argparse
import torch
torch.manual_seed(7)
np.random.seed(5)

MODEL_PATH = '../model/GAN_checkpoint.pth'
img_size = 128
g_conv_dims = 128
n_res_blocks = 10

# set parser parameters
parser = argparse.ArgumentParser('This program allow you to convert human faces into cartoon ones and vise versa.')
parser.add_argument('-i', '--image', type = str, default = 'None', help = 'Image name.')
parser.add_argument('-c', '--convert', type = str, default = 'human', help = 'Indicates to which type of image to convert (human or cartoon). Default value is set to human.')
parser.add_argument('-f', '--filenames', type = str, default = 'image', help = 'Save name used for the resulting images (comparison and new image). If ignored, the default files will be image_comparison.svg and image_result.svg')
arg = parser.parse_args()

# X: human
# Y: cartoon

####################
# Program Functions#
####################

def run():
    if arg.image is not 'None':
        # process image
        image = process_image(arg.image)
        print('Converting ' + arg.image + ' image to ' + arg.convert)
        if arg.convert == 'cartoon':
            # load model
            G_XtoY = load_generator(g_conv_dims, n_res_blocks, 'G_XtoY')
            # set model to eval
            G_XtoY.eval()
            # apply model
            result = G_XtoY(image)
            # plot and save results
            show_results(image, result, arg.filenames)
        elif arg.convert == 'human':
            # load model
            G_YtoX = load_generator(g_conv_dims, n_res_blocks, 'G_YtoX')
            # set model to eval
            G_YtoX.eval()
            # apply model
            result = G_YtoX(image)
            # plot and save results
            show_results(image, result, arg.filenames)
        else:
            raise NotImplementedError('Unknow convert {} parameter. Valid options are: human, cartoon'.format(arg.convert))
    
if __name__ == '__main__':
    run()