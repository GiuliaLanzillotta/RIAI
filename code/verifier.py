import argparse
import torch
import time
from networks import FullyConnected, Conv
from abstract_nets import AbstractFullyConnected, AbstractConv

DEVICE = 'cpu'
INPUT_SIZE = 28


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
    low = torch.max(inputs - eps, torch.tensor(0.0)) # may we should limit this to something very small instaed than 0?
    high = torch.min(inputs + eps, torch.tensor(1.0))
    return inputs, low, high

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
    inputs, low, high = prepare_input_verifier(inputs, eps)
    # 2. Propagate the region across the net
    with torch.no_grad():
        outputs, low, high = net(inputs, low, high)
    # 3. Verify the property 
    # in order to always predict the right label in this perturbation 
    # zone the logits for the right label need to always be 
    # higher than the logits for all the other labels
    verified = sum((low[true_label]>high).int())==9
    end = time.time()
    print("Time to propagate: "+str(round(end-start,3)))
    if verified: return 1
    #4. Backsubstitute if the property is not verified, 
    # otherwise return 
    backsub_order = None
    with torch.no_grad():
        low, high = net.back_sub(inputs, low, high, true_label = true_label, order=backsub_order)
    # for the property to be verified we want all the entries of (y_true - y_j) to be positive
    verified = low.detach().numpy().all()>0
    end = time.time()
    print("Time to backsubstitute: "+str(round(end-start,3)))
    if verified: return 1

    return 0


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

    # TODO: create abstract net here 

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

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(abstract_net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
