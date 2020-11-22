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
    def __init__(self,device, input_size):
        #TODO: initialise lamdas here etc

    def backsub(self):
        return x

    def deepPoly(self,x, high, low, lamda=0):
        x = backsub()
        ub_slope = high/(high-low+1e-6) #upper bound slope with capacity to have high=low=0
        ub_int = (low*high)/(high-low) #intercept of upper bound line
        high = ub_slope*x-ub_int #upper bound from ReLu
        lb_slope = lamda #lower bound line slope
        low = lb_slope*x #lower bound line of ReLu with x coming from backsub
        x = Relu(x) #TODO:correct from pseudocode version
        return (high, low)
        #TODO: NOTE due to timeout do not backsub unless property not proven i.e. low(i)<u(j)<u(i) we want low(i)>u(j)
    def forward(self, x, low, high):
        assert(low <= high).all()
        if ((low < 0) * (high > 0)).any().item():#crossing ReLU outputs True
            '''implement forward version of the DeepPoly'''
        elif high <= 0:
            x = 0
            low = 0
            high = 0
        elif low >= 0:
            x = x
            low = low
            high = high
        return (x, low, high)
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
        #TODO: I believe this is wrong but needs confirming. See slide 23 when affine layer is negative weight max is u-l not u-u


        #now the rest of the layers 
        for i, layer in enumerate(self.layers):
            if type(layer) == AbstractRelu
                x, low, high = AbstractRelu.forward(self, x, low, high)
            else:
                x = layer(x)
                low = layer[i](low)
                high = layer[i](high)
        return(x, low, high)
            pass 
        # TODO: complete the method 
        # TODO: make sure that for each layer we store the boundaries [l,u]
        # as we propagate 
        # TODO: implement backsubstitution
        #TODO: Maybe implement both the areas and for outputs they disagree do something



class AbstractConv(nn.Module):
    """ Abstract version of convolutional model """ 
    pass 