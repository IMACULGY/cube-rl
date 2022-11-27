import torch
import torch.nn as nn
import torch.nn.functional as F

class API_NN(nn.Module):
    def __init__(self):
        # initialize functions
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()

        # initialize layers
        self.l1 = nn.linear(20 * 24, 4096)
        self.l2 = nn.linear(4096, 2048)
        self.l3policy = nn.linear(2048, 512)
        self.l3value = nn.linear(2048, 512)
        self.outpolicy = nn.linear(512,12)
        self.outvalue = nn.linear(512,1)

    def forward(self,x):
        # calculate up to the second hidden layer (2048)
        current = torch.FloatTensor(x)
        current = self.elu(self.l1(current))
        current = self.elu(self.l2(current))
        # calculate policy
        policy = self.elu(self.l3policy(current))
        policy = self.softmax(self.outpolicy(policy))
        # calculate value
        value = self.elu(self.l3value(current))
        value = self.outvalue(value)

        return policy, value

# supervised training for NN with input [X,Y]
def TrainNN(net, x, y):
    # define losses and optimizer
    celoss = None
    mse = None
    opt = None

    EPOCHS = 10000

    for _ in range(EPOCHS):
        opt.zero_grad()

        # forward propagation
        pred = net(x)
        
        # compute loss

        # backward propagation

    return net

# initialize weights of layer m using Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# API algorithm (autodidactic iteration)
def API(num_iter, ):
    # initialize neural net
    net = API_NN()
    net.apply(init_weights)

    for i in num_iter:



    