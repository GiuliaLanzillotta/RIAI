""" 
Here we define the classes that we will use to represent abstract networks. 
An abstract network is a copy of a concrete network implementing abstract transformers 
for each of the operations defined in the concrete network.

The specifics of the transformers depend on the relaxation in use. In our case 
this is polytopes.

This file mimics the structure of the networks.py file. 
"""
from .networks import Normalization
from torch import nn

class AbstractRelu(nn.Module):
    """ Abstract version of ReLU layer """
    pass 

class AbstractFullyConnected(nn.Module):
    """ Abstract version of fully connected network """
    def __init__(self, device, input_size, fc_layers):
        super(AbstractFullyConnected).__init__()
        # TODO: make sure that normalisation is not 
        # affecting the verification 
        layers = [Normalization(device), 
                    nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.AbstractRelu()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)
    

    def forward(self, x, low, high):
        """
        Propagation of abstract area through the network. 
        Parameters: 
        - x: input 
        - low: lower bound on input perturbation (epsilon)
        - high: upper bound on //   // 
        note: all the input tensors have shape (1,1,28,28)

        """
        # propagate normally through the first two layers 
        x = self.layers[0](x) # normalization 
        x = self.layers[1](x) # flattening 
        # we can safely pass the perturbation boundaries through 
        # normalization and flattening (affine transformation is exact)
        low = self.layers[0](low)
        low = self.layers[1](low)
        high = self.layers[0](high)
        high = self.layers[1](high)

        #now the rest of the layers 
        for i, layer in enumerate(self.layers):
            pass 
        # TODO: complete the method 
        # TODO: make sure that for each layer we store the boundaries [l,u]
        # as we propagate 
        # TODO: implement backsubstitution



class AbstractConv(nn.Module):
    """ Abstract version of convolutional model """ 
    pass 