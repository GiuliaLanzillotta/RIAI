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

DEVICE = 'cpu'

class AbstractLinear(nn.Module):
    """ Abstract version of linear layer """ 
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.layer.requires_grad_(False)
    
    @staticmethod
    def forward_boxes(weights, bias, low, high):
        """
        Implements swapping of lower and higher bounds 
        where the weights are negative and computes the 
        forward pass of box bounds. """

        mask_neg = (weights < 0).int()
        mask_pos = (weights >= 0).int()
        weight_neg = torch.multiply(mask_neg, weights)
        weight_pos = torch.multiply(mask_pos, weights)
        low_out = (torch.matmul(high, weight_neg.t()) + torch.matmul(low, weight_pos.t()) + bias)
        high_out = (torch.matmul(low, weight_neg.t()) + torch.matmul(high, weight_pos.t()) + bias)

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
    def __init__(self, input_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.size = input_dim
        self.lamda = torch.zeros(input_dim)
        self.is_lamda_active = False


    def activate_lamda_update(self):
        self.lamda = torch.nn.Parameter(self.lamda, requires_grad=True)
        self.is_lamda_active = True

    @staticmethod
    def val_lamda(low, high):
        """ Implementing minimum area logic"""
        if low ** 2 > high ** 2: lamda = 0
        else:lamda = 1
        return lamda

    def deepPoly(self, high, low, i):
        # compute the upper bound slope and intercept
        ub_slope = high/(high-low) #upper bound slope with capacity to have high=low=0
        ub_int = -(low*high)/(high-low) #intercept of upper bound line
        # save weight and biases for lower and upper bounds
        self.weight_high[i,i] = ub_slope
        self.bias_high[i] = ub_int
        if not self.is_lamda_active:
            self.lamda[i] = self.val_lamda(low, high)
        self.weight_low[i, i] = self.lamda[i]

    def forward(self, x, low, high):

        input_size = x.size()[0]

        # Initialise the matrices
        self.weight_low = torch.eye(input_size, input_size)
        self.bias_low = torch.zeros(input_size)
        self.weight_high = torch.eye(input_size, input_size)
        self.bias_high = torch.zeros(input_size)

        for i in range(input_size):
            if ((low[i] < 0) * (high[i] > 0)): #crossing ReLU outputs True
                '''implement forward version of the DeepPoly'''
                self.deepPoly(high[i], low[i], i) # modify weights
            elif high[i] <= 0:
                self.weight_high[i, i] = 0
                self.weight_low[i, i] = 0
            else: pass
            # note: if low >=0 we have not done anything,
            # so we can just return the input!
        # compute lower and upper bounds
        # Build the output
        x_out = self.relu(x)
        high_out = torch.matmul(self.weight_high,high) + self.bias_high
        low_out = torch.matmul(self.weight_low,low) + self.bias_low

        return x_out, low_out, high_out

class AbstractFullyConnected(nn.Module):
    """ Abstract version of fully connected network """
    def __init__(self, device, input_size, fc_layers):
        super().__init__()
        layers = [Normalization(device), 
                    nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            with torch.no_grad():
                layers += [AbstractLinear(prev_fc_size, fc_size).to(DEVICE)]
            if i + 1 < len(fc_layers):
                layers += [AbstractRelu(fc_size).to(DEVICE)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def load_weights(self, net):
        for i, layer in enumerate(net.layers):
            if type(layer) == nn.Linear:
                self.layers[i].layer.weight = layer.weight
                self.layers[i].layer.weight.requires_grad_(False)
                self.layers[i].layer.bias = layer.bias
                self.layers[i].layer.bias.requires_grad_(False)

    def back_sub_layers(self, layer_index, size_input):
        """ Implements backsubstitution up to the layer layer_index
        """
        #TODO: make this function faster by saving preious layers backsub results
        #TODO: remove duplicated code and summarise backsub into a single function
        low = self.lows[0]
        high = self.highs[0]

        W_low = torch.eye(size_input, size_input)
        W_high = torch.eye(size_input, size_input)
        bias_high = torch.zeros(size_input)
        bias_low = torch.zeros(size_input)

        for layer in reversed(self.layers[2:layer_index+1]):
            if type(layer) == AbstractLinear:
                W_prime_high = layer.layer.weight
                b_prime_high = layer.layer.bias
                W_prime_low = layer.layer.weight
                b_prime_low = layer.layer.bias
                bias_high += torch.matmul(W_high, b_prime_high)
                bias_low += torch.matmul(W_low, b_prime_low)
                W_high = torch.matmul(W_high, W_prime_high)
                W_low = torch.matmul(W_low, W_prime_low)
            elif type(layer) == AbstractRelu:
                W_prime_low = layer.weight_low
                W_prime_high = layer.weight_high
                b_prime_high = layer.bias_high
                W_high, delta_bias_high = self.back_sub_relu(W_high, W_prime_high, W_prime_low, bias_high=b_prime_high)
                W_low, delta_bias_low = self.back_sub_relu(W_low, W_prime_high, W_prime_low, bias_high=b_prime_high, high=False)
                bias_high += delta_bias_high
                bias_low += delta_bias_low
            else:
                raise Exception("Unknown layer in the forward pass ")

        # finally computing the forward pass on the input ranges
        # note: no bias here (all the biases were already included in W)
        low_out, _ = AbstractLinear.forward_boxes(W_low, bias_low, low, high)
        _, high_out = AbstractLinear.forward_boxes(W_high, bias_high, low, high)

        return low_out, high_out

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

        self.lows = []
        self.highs = []
        self.activations = []

        self.lows+=[low]
        self.highs+=[high]
        self.activations+=[x]
        #now the rest of the layers
        for i, layer in enumerate(self.layers):
            if i in [0,1]: continue # skipping the ones we already computed 
            # no need to distinguish btw layers as they have same signature now
            x, low, high = layer(x, low, high)
            if type(layer)==AbstractLinear:
                # note: even though we backsubstitute at each affine,
                # there is still a dependency on the lamdas
               low, high = self.back_sub_layers(layer_index=i,size_input=x.size()[0])
            self.lows+=[low]
            self.highs+=[high]
            self.activations+=[x]


        return x, low, high

    def clamp_lamdas(self):
        """ Clamp the value of the lamdas for all the ReLus
        in the net to the range [0,1]"""
        for i, layer in enumerate(self.layers):
            if type(layer) == AbstractRelu:
                new_lamda = layer.lamda.clone()
                new_lamda.clamp_(min=0, max=1)
                if i == len(self.layers)-2:
                    print(new_lamda)
                # new_lamda[new_lamda<0.5] = 0
                # new_lamda[new_lamda >= 0.5] = 1
                # print(new_lamda)
                layer.lamda = torch.nn.Parameter(new_lamda, requires_grad=True)
                #print(layer.lamda)

    def activate_lamdas(self):
        """ Clamp the value of the lamdas for all the ReLus
        in the net to the range [0,1]"""
        for layer in self.layers:
            if type(layer) == AbstractRelu:
                layer.activate_lamda_update()

    def back_sub_relu(self, back_sub_matrix, relu_high_matrix, relu_low_matrix, bias_high, high=True):
        """ Computes matrix multiplication for backsubstitution
        when passing through a relu layer """
        out_dim, in_dim = back_sub_matrix.size()
        #initialise everything to 0
        output_matrix = torch.zeros_like(back_sub_matrix)
        bias_vector = torch.zeros(back_sub_matrix.size()[0])
        # now we want to go into each entry in the back_sub matrix
        # and multiply it by the respective relu weight
        for j in range(in_dim):# for each column of the matrix
            relu_high_factor = relu_high_matrix[j,j]
            relu_low_factor = relu_low_matrix[j,j]
            if relu_high_factor !=0: # otherwise we keep the default
                for i in range(out_dim): # for each row in the column
                    entry = back_sub_matrix[i,j]
                    if entry>0:
                        if high:
                            output_matrix[i,j] = entry*relu_high_factor
                            bias_vector[i] += entry*bias_high[j]
                        else: output_matrix[i,j] = entry*relu_low_factor
                    else:
                        if high: output_matrix[i,j] = entry*relu_low_factor
                        else:
                            output_matrix[i,j] = entry*relu_high_factor
                            bias_vector[i] += entry * bias_high[j]
        return output_matrix, bias_vector

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
        for layer in reversed(self.layers[-(order-1):]): # order = layers -1 --> order -1 = layers -2 --> skipping first two layers
            if type(layer) == AbstractLinear:
                W_prime_high = layer.layer.weight
                b_prime_high = layer.layer.bias
                W_prime_low = layer.layer.weight
                b_prime_low = layer.layer.bias
                bias_high += torch.matmul(W_high, b_prime_high)
                bias_low += torch.matmul(W_low, b_prime_low)
                W_high = torch.matmul(W_high, W_prime_high)
                W_low = torch.matmul(W_low, W_prime_low)

            elif type(layer) == AbstractRelu:
                W_prime_low = layer.weight_low
                W_prime_high = layer.weight_high
                b_prime_high = layer.bias_high
                W_high, delta_bias_high = self.back_sub_relu(W_high, W_prime_high, W_prime_low, bias_high = b_prime_high)
                W_low, delta_bias_low = self.back_sub_relu(W_low, W_prime_high, W_prime_low, bias_high = b_prime_high, high=False)
                bias_high += delta_bias_high
                bias_low += delta_bias_low

            else:
                raise Exception("Unknown layer in the forward pass ")


        # finally computing the forward pass on the input ranges
        # note: no bias here (all the biases were already included in W)
        low_out, _ = AbstractLinear.forward_boxes(W_low, bias_low, low, high)
        _, high_out = AbstractLinear.forward_boxes(W_high, bias_high, low, high)

        return low_out, high_out

class AbstractConvLayer(nn.Module):
    def __init__(self, prev_channels, n_channels, kernel_size, stride, padding, input_dim):
        super().__init__()
        self.layer = nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding)

        self.prev_channels = prev_channels
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_dim = input_dim

    def cast_to_affine(self):
        """ Converts a convolutional layer into an affine layer
        by calculating a weight matrix and bias equivalent to the convolution
        kernel """
        # First of all we should figure out the output dimensions
        out_img_dim = self.input_dim//2 # checked by hand. Note: this formula only holds for our nets
        self.vectorised_input_dim = (self.input_dim**2)*self.prev_channels
        self.vectorised_output_dim = (out_img_dim**2)*self.n_channels
        self.weight = torch.zeros(self.vectorised_output_dim, self.vectorised_input_dim, requires_grad=False)
        self.bias = torch.zeros(self.vectorised_output_dim, requires_grad=False)
        kernel = self.layer.weight
        kernel_bias = self.layer.bias
        # okay now let's fill these matrices
        # note: we have to take into account also the stride!
        counter = 0
        for k in range(self.n_channels):
            for j in range(self.kernel_size - self.padding, self.input_dim + self.padding + 1, self.stride):  # move down the rows
                for i in range(self.kernel_size - self.padding, self.input_dim + self.padding + 1, self.stride):  # move right in the columns
                    # initialise the vector for the row
                    # we will have prev_channels of these vectors
                    vec = torch.zeros(self.input_dim ** 2 * self.prev_channels, requires_grad=False)
                    #print(" i ", i, "| j ", j, "| k ", k)
                    for c in range(self.prev_channels):
                        for j_prim in range(max(0, j - self.kernel_size), min(j, self.input_dim)):
                            # j_prim*self.input_dim = shift in the rows of the original image (each row is afer self.input_dim entries)
                            # c * self.input_dim**2 = shift in the channels of the original image (each new channel after self.input_dim*2 entries)
                            start = j_prim * self.input_dim + c * self.input_dim ** 2 + max(i - self.kernel_size,
                                                                                  0)  # i is the last column we are touching
                            end = min(i + j_prim * self.input_dim + c * self.input_dim ** 2,
                                      c * self.input_dim ** 2 + (j_prim + 1) * self.input_dim)
                            kernel_row = min(j - j_prim, self.kernel_size)  # j is the last row we are touching
                            kernel_col = min(i, self.kernel_size)
                            num_columns_to_use = end - start
                            if num_columns_to_use < self.kernel_size < i:
                                vec[start:end] = kernel[k, c, -kernel_row, -kernel_col:-kernel_col + num_columns_to_use]
                            else:
                                vec[start:end] = kernel[k, c, -kernel_row, -kernel_col:]
                    self.bias[counter] = kernel_bias[k]
                    self.weight[counter, :] = vec

                    counter += 1

class AbstractConv(nn.Module):
    """ Abstract version of convolutional model """

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super().__init__()
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
                AbstractConvLayer(prev_channels, n_channels, kernel_size, stride=stride, padding=padding, input_dim = img_dim),
                AbstractRelu(img_dim // stride)
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [AbstractLinear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [AbstractRelu(fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def load_weights(self, net):
        for i, layer in enumerate(net.layers):
            if type(layer) in [nn.Conv2d, nn.Linear]:
                self.layers[i].layer.weight = layer.weight
                self.layers[i].layer.bias = layer.bias

    def make_fully_connected(self, device):
        """ Returns the fully connected equivalent of the current convolutional net"""
        layers_dims = []
        layers_weights = []
        layers_biases = []
        for layer in self.layers:
            if type(layer)==AbstractConvLayer:
                layer.cast_to_affine() # first cast
                weight = layer.weight # then copy
                bias = layer.bias
            elif type(layer)==AbstractLinear:
                weight = layer.layer.weight
                bias = layer.layer.bias
            else : continue

            layers_dims += [weight.size()[0]]
            layers_weights += [weight]
            layers_biases += [bias]

        FCnet = AbstractFullyConnected(device, self.input_size, layers_dims).to(device)
        # finally load the weights
        cnt = 0
        for layer in FCnet.layers:
            if type(layer) == AbstractLinear:
                layer.layer.weight = torch.nn.Parameter(layers_weights[cnt], requires_grad=False)
                layer.layer.bias = torch.nn.Parameter(layers_biases[cnt], requires_grad=False)
                cnt +=1

        return FCnet



