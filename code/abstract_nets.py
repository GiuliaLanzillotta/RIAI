""" 
Here we define the classes that we will use to represent abstract networks. 
An abstract network is a copy of a concrete network implementing abstract transformers 
for each of the operations defined in the concrete network.

The specifics of the transformers depend on the relaxation in use. In our case 
this is polytopes.

This file mimics the structure of the networks.py file. 
"""
from networks import Normalization
from torch import nn
import torch

class AbstractLinear(nn.Module):
    """ Abstract version of linear layer """ 
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)
    
    @staticmethod
    def forward_boxes(weights, bias, low, high):
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
            low_out[i] = torch.matmul(low_input, w) + bias[i]
            high_out[i] = torch.matmul(high_input, w) + bias[i]
        
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
    def __init__(self, lamda=0.0):
        super().__init__()
        self.lamda = lamda
        self.relu = nn.ReLU()

    def deepPoly(self, high, low, i):
        # compute the upper bound slope and intercept
        ub_slope = high/(high-low) #upper bound slope with capacity to have high=low=0
        ub_int = -(low*high)/(high-low) #intercept of upper bound line
        # save weight and biases for lower and upper bounds
        self.weight_high[i,i] = ub_slope
        self.bias_high[i] = ub_int
        self.weight_low[i, i] = self.lamda

    def forward(self, x, low, high):

        input_size = x.size()[0]

        # Initialise the matrices
        self.weight_low = torch.eye(input_size, input_size)
        self.bias_low = torch.zeros(input_size)
        self.weight_high = torch.eye(input_size, input_size)
        self.bias_high = torch.zeros(input_size)

        print(input_size)
        for i in range(input_size):
            if ((low[i] < 0) * (high[i] > 0)): #crossing ReLU outputs True
                '''implement forward version of the DeepPoly'''
                self.deepPoly(high[i], low[i], i) # modify weights
            elif high[i] <= 0:
                self.weight_high[i, i] = 0
                self.weight_low[i, i] = 0
            else:
                pass
            # note: if low >=0 we have not done anything,
            # so we can just return the input!
        # compute lower and upper bounds
        # Build the output
        print(self.weight_high)
        print(self.weight_low)
        x_out = self.relu(x)
        high_out = torch.matmul(self.weight_high,high) + self.bias_high
        low_out = torch.matmul(self.weight_low,low) + self.bias_low

        return x_out, low_out, high_out

class AbstractReluConv(nn.Module):
    def __init__(self, lamda=0.0):
        super().__init__()
        self.lamda = lamda
        self.relu = nn.ReLU()

class AbstractFullyConnected(nn.Module):
    """ Abstract version of fully connected network """
    def __init__(self, device, input_size, fc_layers):
        super().__init__()
        layers = [Normalization(device), 
                    nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [AbstractLinear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [AbstractRelu(lamda=0.0)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)
        self.lows = []
        self.highs = []
        self.activations = []
    

    def load_weights(self, net):
        for i, layer in enumerate(net.layers):
            if type(layer) == nn.Linear:
                self.layers[i].layer.weight = layer.weight
                self.layers[i].layer.bias = layer.bias


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
        x = self.layers[1](x).squeeze() # flattening and removing extra dimension
        # we can safely pass the perturbation boundaries through 
        # normalization and flattening (affine transformation is exact)
        low = self.layers[0](low)
        low = self.layers[1](low).squeeze()
        high = self.layers[0](high)
        high = self.layers[1](high).squeeze()


        self.lows+=[low]
        self.highs+=[high]
        self.activations+=[x]
        #now the rest of the layers 
        for i, layer in enumerate(self.layers):
            if i in [0,1]: continue # skipping the ones we already computed 
            # no need to distinguish btw layers as they have same signature now
            x, low, high = layer(x, low, high)
            self.lows+=[low]
            self.highs+=[high]
            self.activations+=[x]

        return x, low, high

    
    def back_sub(self, true_label, order=None):
        """ Implements backsubstitution 
        true_label (int): index (0 to 9) of the right label - used in the last step of backsubstitution
        order (int): defines number of layers to backsubstitute starting from the output.  
        """
        if order is None: order = len(self.activations) # example: 10 layers, 9 actual lows and highs, 1 for the input, 8 for the rest of the layers
        low = self.lows[-order]
        high = self.highs[-order]


        num_classes = 10 # we will start from the output
        bias_high = torch.zeros(num_classes-1)
        bias_low = torch.zeros(num_classes-1)
        #bias_high = torch.zeros(num_classes)
        #bias_low = torch.zeros(num_classes)


        # First, we insert the affine layer corresponding to the substractions
        # employed by the verifier to check the correctness of the prediction
        # output_j = logit_i - logit_j, where i is the true_label
        W_substract = torch.eye(num_classes-1, num_classes-1)*(-1)
        W_substract = torch.cat([W_substract[:, 0:true_label],
                                 torch.ones(num_classes-1, 1),
                                 W_substract[:, true_label:num_classes]], 1) # inserting the column of ones for the true label
        # now cumulating the last operation
        W_low = W_substract.clone()
        W_high = W_substract.clone()
        #W_low = torch.eye(num_classes)
        #W_high = torch.eye(num_classes)
        for layer in reversed(self.layers[-(order-1):]): # order = layers -1 --> order -1 = layers -2 --> skipping first two layers
            if type(layer) == AbstractLinear: 
                W_prime_high = layer.layer.weight 
                b_prime_high = layer.layer.bias
                W_prime_low = layer.layer.weight 
                b_prime_low = layer.layer.bias
            elif type(layer) == AbstractRelu:
                W_prime_low = layer.weight_low
                b_prime_low = layer.bias_low
                W_prime_high = layer.weight_high
                b_prime_high = layer.bias_high
            else: 
                raise Exception("Unknown layer in the forward pass ")
            bias_high += torch.matmul(W_high, b_prime_high)
            W_high = torch.matmul(W_high, W_prime_high)
            bias_low += torch.matmul(W_low, b_prime_low)
            W_low = torch.matmul(W_low,W_prime_low)


        # finally computing the forward pass on the input ranges
        # note: no bias here (all the biases were already included in W)
        low_out, _ = AbstractLinear.forward_boxes(W_low, bias_low, low, high)
        _, high_out = AbstractLinear.forward_boxes(W_high, bias_high, low, high)

        return low_out, high_out


class AbstractConvLayer(nn.Module):
    def __init__(self, prev_channels, n_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Conv2d(prev_channels, n_channels, kernel_size, stride, padding)

        self.prev_channels = prev_channels
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @staticmethod
    def forward_image_boxes(low, high, weight, biases):
        low = low.squeeze()
        high = high.squeeze()
        weight = weight.view(weight.size(0), -1).t()
        mask_neg = (weight < 0).int()
        mask_pos = (weight < 0).int()
        weight_neg = torch.multiply(mask_neg, weight).t()
        weight_pos = torch.multiply(mask_pos, weight).t()
        low_out = (torch.matmul(high,weight_neg)+torch.matmul(low, weight_pos) + biases).t()
        high_out = (torch.matmul(low, weight_neg) + torch.matmul(high, weight_pos) +biases).t()

        # quick check here
        assert (low_out <= high_out).all(), "Error with the box bounds: low>high"
        return low_out, high_out

    def forward(self, x, low, high):
        w = self.layer.weight
        b = self.layer.bias
        # Handmade conv
        low = torch.nn.functional.unfold(low, self.kernel_size, 1, self.padding, self.stride).transpose(1, 2)
        high = torch.nn.functional.unfold(high, self.kernel_size, 1, self.padding, self.stride).transpose(1, 2)
        low, high = self.forward_image_boxes(low, high, w, b)
        x = self.layer(x)
        low = low.view(x.size)
        high = high.view(x.size)

        return x, low, high


class AbstractConv(nn.Module):
    """ Abstract version of convolutional model """

    def init(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(AbstractConv, self).init()
        self.lows = []
        self.highs = []
        self.activations = []

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                AbstractConvLayer(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                AbstractReluConv(device, kernel_size),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [AbstractLinear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [AbstractRelu(device, fc_size)]
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
        x = self.layers[0](x)  # normalization
        low = self.layers[0](low)
        high = self.layers[0](high)
        self.lows += [low]
        self.highs += [high]
        self.activations += [x]
        # now the rest of the layers
        for i, layer in enumerate(self.layers):
            if i in [0]: continue  # skipping the ones we already computed
            # no need to distinguish btw layers as they have same signature now
            if type(layer) == nn.Flatten:
                x = layer(x).squeeze()
                low = layer(low).squeeze()
                high = layer(high).squeeze()
                continue
            x, low, high = layer(x, low, high)
            self.lows += [low]
            self.highs += [high]
            self.activations += [x]

        return x, low, high