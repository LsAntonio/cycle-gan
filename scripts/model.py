#############
# Libraries #
#############

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(7)

MODEL_PATH = '../model/CGAN_checkpoint.pth'

####################
# Helper Functions #
####################

class ResidualBlock(nn.Module):
    """
    Defines a residual block.
    This adds an input x to a convolutional layer (applied to x) with the same size input and output.
    These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # number of inputs
        self.conv_dim = conv_dim
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        self.conv1 = nn.Conv2d(conv_dim, conv_dim, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(conv_dim)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(conv_dim)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out = self.conv1(x)
        out = self.batch_norm1(x)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        
        return x + out
    
def deconv(in_channels, out_channels, kernel_size, layer_type, stride=2, padding=1, negative_slope = 0.2):
    """
    Creates a transpose convolutional layer, with optional batch normalization.
    
    params
    ------
    in_channels -- Input channels
    out_channels -- Ouput channels
    kernel_size -- Kernel size used in the conv layers
    layer_type -- Layer type. It could be normal or final
    stride -- Stride of the conv layers.
    padding -- Padding used.
    negative_slope -- Negative slope rate for the ReLU activation.
    
    returns
    -------
    A set of transpose conv layers.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if layer_type == 'normal':
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope))
        return nn.Sequential(*layers)
      
    elif layer_type == 'final':
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    else:
        raise NotImplementedError('Unknown layer type: {}'.format(layer_type))

def enconv(in_channels, out_channels, kernel_size, stride=2, padding=1, negative_slope = 0.2):
    """
    Creates a transpose convolutional layer, with optional batch normalization.
    
    params
    ------
    in_channels -- Input channels
    out_channels -- Ouput channels
    kernel_size -- Kernel size used in the conv layers
    stride -- Stride of the conv layers.
    padding -- Padding used.
    negative_slope -- Negative slope rate for the ReLU activation.
    
    returns
    -------
    A set of convolutional layers.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(negative_slope))
    return nn.Sequential(*layers)



class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=8):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        self.conv1 = enconv(3, conv_dim, 4, stride=2, padding=1, negative_slope = 0.2)
        self.conv2 = enconv(conv_dim, conv_dim * 2, 4, stride=2, padding=1, negative_slope = 0.2)
        self.conv3 = enconv(conv_dim * 2, conv_dim * 4, 4, stride=2, padding=1, negative_slope = 0.2)
        
        # 2. Define the resnet part of the generator
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim * 4))
          
        self.res_block = nn.Sequential(*res_layers)
        # 3. Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim * 4, conv_dim * 2, 4, 'normal', stride=2, padding=1, negative_slope = 0.2)
        self.deconv2 = deconv(conv_dim * 2, conv_dim, 4, 'normal', stride=2, padding=1, negative_slope = 0.2)
        self.deconv3 = deconv(conv_dim, 3, 4, 'final', stride=2, padding=1, negative_slope = 0.2)
        
    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_block(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x