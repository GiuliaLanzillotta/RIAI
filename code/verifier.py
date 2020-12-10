import argparse
import torch
import time
from networks import FullyConnected, Conv
from abstract_nets import AbstractFullyConnected, AbstractConv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Warning we get: (torch 1.7 on) .../torch/autograd/__init__.py:132: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.

DEVICE = 'cpu'
INPUT_SIZE = 28
NUM_EPOCHS = 5
LEARNING_RATE = 10e-3
MOMENTUM = 0.9

class LamdaLoss(torch.nn.Module):
    """ Custom loss function to optimise the lamdas"""
    def __init__(self):
        super(LamdaLoss, self).__init__()

    @staticmethod
    def forward(last_low, last_high, right_class):
        """ Compute the loss on the lamdas as the sum of the width of the
        last layers' neurons' bounds"""
        assert (last_high>=last_low).all(), "Error with the box bounds: low>high"
        loss = max(torch.cat([last_high[0:right_class], last_high[right_class+1:]])) - last_low[right_class]
        return loss

def prepare_input_verifier(inputs, eps):
    """ 
    This function computes the input to the verifier. 
    Given an input image and a noise level it computes the range 
    of values for every pixel. 
    From the task description: 
    'Note that images have pixel intensities between 0 and 1, 
    e.g. if perturbation is 0.2 and pixel has value 0.9 then you only 
    have to verify range [0.7, 1.0] for intensities of this pixel, 
    instead of [0.7, 1.1]'
    
    """
    # inputs has shape (1,1,28,28)
    # hence also eps has the same shape
    low = torch.max(inputs - eps, torch.tensor(0.0)).to(DEVICE) # may we should limit this to something very small instead than 0?
    high = torch.min(inputs + eps, torch.tensor(1.0)).to(DEVICE)
    return inputs, low, high

def update_lamdas(inputs, low_orig, high_orig, net, right_class):
    """ Wrapping function for all the operation necessary
    to make an optimization step for the lamdas"""

    outputs, low, high = net(inputs, low_orig, high_orig)
    loss = LamdaLoss().to(DEVICE)
    loss_value = loss(low, high, right_class)
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return outputs, low, high



def analyze(net, inputs, eps, true_label):
    """
        This function should run the DeepPoly relaxation on the L infinity 
        ball of radius epsilon around the input and verify whether the net 
        would always output the right label. 

        [input +-eps] --> [y_true-label > y_i] for all i != true-label

        Arguments
        ---------
        net: (nn.Module) - either instance of AbstractFullyConnected or AbstractConv  
        inputs: (FloatTensor) - shape (1, 1, 28, 28)
        eps: (float) - noise level
        true_label: (int) - label from 0 to 9  

        Returns
        --------
        (bool) - True if the property is verified
    """
    start = time.time()
    # 1. Define the input box - the format should be defined by us 
    # as it will be used by our propagation function. 
    inputs, low_orig, high_orig = prepare_input_verifier(inputs, eps)
    # 2. Propagate the region across the net
    outputs, low, high = net(inputs, low_orig, high_orig)
    pred_label = outputs.max(dim=0)[1].item()
    assert pred_label == true_label
    # 3. Verify the property 
    # in order to always predict the right label in this perturbation 
    # zone the logits for the right label need to always be 
    # higher than the logits for all the other labels
    verified = sum((low[true_label]>high).int())==9
    end = time.time()
    print("Time to propagate: "+str(round(end-start,3)))
    if verified: return verified

    # 4. Backsubstitute if the property is not verified, otherwise return
    # net.reset_lamdas()
    backsub_order = None
    with torch.no_grad():
        low_bs, high_bs = net.back_sub(true_label=true_label, order=backsub_order)
    # for the property to be verified we want all the entries of (y_true - y_j) to be positive
    verified = (low_bs.detach().numpy() > 0).all()
    end = time.time()
    print("Time to backsubstitute: " + str(round(end - start, 3)))
    if verified: return verified


    # 5. Update the lamdas to optimise our loss and try to verify again
    start = time.time()
    print("Optimising the lamdas...")
    for epoch in range(NUM_EPOCHS):
        print("Epoch "+str(epoch))
        net.activate_lamdas()
        outputs, low, high = update_lamdas(inputs, low_orig, high_orig, net, true_label)
        pred_label = outputs.max(dim=0)[1].item()
        assert pred_label == true_label     #check that only the lamdas have been changed.
        verified = sum((low[true_label] > high).int()) == 9
        if verified: return verified
        # note: the lamdas have to be between 0 and 1, hence we
        # cannot simply update them with a gradient descent step
        # - we also need to project back to the [0, 1] box
        net.clamp_lamdas()
        end = time.time()
        print("Time: " + str(round(end - start, 3)))
    end = time.time()
    print("Second time to propagate: "+str(round(end-start,3)))

    # 6. Backsubstitute if the property is not verified, otherwise return
    #net.reset_lamdas()
    backsub_order = None
    with torch.no_grad():
        low, high = net.back_sub(true_label = true_label, order=backsub_order)
    # for the property to be verified we want all the entries of (y_true - y_j) to be positive
    verified = (low.detach().numpy()>0).all()
    end = time.time()
    print("Time to backsubstitute: "+str(round(end-start,3)))
    return verified


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])


    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE,[100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
        abstract_net = AbstractConv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
        abstract_net = AbstractConv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
        abstract_net = AbstractConv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False


    # here we are loading the pre-trained net weights 
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))
    abstract_net.load_weights(net)
    if type(abstract_net)==AbstractConv:
        #cast it to fully connected
        print("Converting the network to fully connected...")
        abstract_net = abstract_net.make_fully_connected(DEVICE).to(DEVICE)
        print("Conversion done")

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(abstract_net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')

    print("-"*20)


if __name__ == '__main__':
    main()
