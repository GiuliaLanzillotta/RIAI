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
import torch

class AbstractLinear(nn.Module):
    """ Abstract version of linear layer """ 
    def __init__(self, input_size, output_size):
        super(AbstractLinear).__init__()
        self.layer = nn.Linear(input_size, output_size)
    
    def forward_boxes(self, weights, bias, low, high):
        """
        Implements swapping of lower and higher bounds 
        where the weights are negative and computes the 
        forward pass of box bounds. 
        """
        low_out = torch.zeros(weights.size()[0])
        high_out = torch.zeros(weights.size()[0])

        # weights is a matrix with shape [in, out]
        for i, w in enumerate(weights): 
            # now w is a vector of size [in]
            # determine the swapping factor 
            mask_neg = (w<0).int()
            mask_pos = (w>=0).int()
            low_input = mask_neg*high + mask_pos*low
            high_input = mask_pos*high + mask_neg*low
            # ready to make the forward pass 
            low_out[i] = torch.matmul(low_input, w) + bias
            low_out[i] = torch.matmul(high_input, w) + bias
        
        # quick check here 
        assert (low_out <= high_out).all(), "Error with the box bounds: low>high"
        return low_out, high_out

    def forward(self, x, low, high): 
        """ 
        Specific attention must be payed to the transformation 
        of box bounds. When working with negative weights low and 
        high bound must be swapped. 
        """
        x = self.layer(x)
        weights = self.layer.weight
        bias = self.layer.bias
        low, high = self.forward_boxes(weights, bias, low, high) 
        return x, low, high

class AbstractRelu(nn.Module):
    """ Abstract version of ReLU layer """
    def __init__(self,device, input_size):
        #TODO: initialise lamdas here etc
        pass

    def backsub(self):
        pass

    def deepPoly(self, x, high, low, lamda=0):
        x = backsub()
        ub_slope = high/(high-low+1e-6) #upper bound slope with capacity to have high=low=0
        ub_int = (low*high)/(high-low) #intercept of upper bound line
        high = ub_slope*x-ub_int #upper bound from ReLu
        lb_slope = lamda #lower bound line slope
        low = lb_slope*x #lower bound line of ReLu with x coming from backsub
        x = nn.Relu(x) #TODO:correct from pseudocode version
        return (high, low)
        #TODO: NOTE due to timeout do not backsub unless property not proven i.e. low(i)<u(j)<u(i) we want low(i)>u(j)
    
    def forward(self, x, low, high):
        if ((low < 0) * (high > 0)).any().item(): #crossing ReLU outputs True
            '''implement forward version of the DeepPoly'''
        elif high <= 0: x, low, high  = 0, 0, 0
        # note: if low >=0 we have not done anything, 
        # so we can just return the input! 
        return x, low, high
     

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
            layers += [nn.AbstractLinear(prev_fc_size, fc_size)]
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
        #TODO: I believe this is wrong but needs confirming. 
        # See slide 23 when affine layer is negative weight max is u-l not u-u

        #now the rest of the layers 
        for i, layer in enumerate(self.layers):
            if i in [0,1]: continue # skipping the ones we already computed 
            # no need to distinguish btw layers as they have same signature now
            x, low, high = layer(x, low, high) 
        return x, low, high

        # TODO: implement backsubstitution
        # TODO: Maybe implement both the areas and for outputs they disagree do something



class AbstractConv(nn.Module):
    """ Abstract version of convolutional model """ 
    pass 