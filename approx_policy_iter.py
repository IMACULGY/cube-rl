import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

from Cube import Cube
from encode_cube import encode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class API_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize functions
        self.elu = nn.ELU()
        self.softmax = nn.Softmax()

        # initialize layers
        self.l1 = nn.Linear(20 * 24, 4096)
        self.l2 = nn.Linear(4096, 2048)
        self.l3policy = nn.Linear(2048, 512)
        self.l3value = nn.Linear(2048, 512)
        self.outpolicy = nn.Linear(512,12)
        self.outvalue = nn.Linear(512,1)

    def forward(self,x):
        # calculate up to the second hidden layer (2048)
        current = torch.tensor(x).to(device).to(torch.float)
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
def TrainNN(net, x, y, weights, EPOCHS=10):
    # initialize tensors
    x = torch.tensor(x).to(device)
    y1 = torch.stack([a[0] for a in y],dim=0).to(device)
    y2 = torch.FloatTensor([[a[1]] for a in y]).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # print(weights)

    # define losses and optimizer
    celoss = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    opt = torch.optim.RMSprop(net.parameters(), lr=0.0001)
    sumlosses = 0

    # progress bar!
    bar = tqdm.trange(EPOCHS, desc="epoch")


    for _ in bar:
        opt.zero_grad()

        # forward propagation
        pred = net(x)
        
        # compute loss
        l1 = celoss(y1, pred[0])
        l2 = mseloss(y2, pred[1])
        sumlosses = sum([l1, l2])
        # apply sample weighting
        sumlosses = sumlosses * weights
        sumlosses = sumlosses.mean()

        bar.set_description(f"Loss = {sumlosses.item()}")

        # backward propagation
        sumlosses.backward()
        opt.step()

    return sumlosses.item()

# initialize weights of layer m using Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# API algorithm (autodidactic iteration)
def API(env, num_iter=10000, load=False, loadPath="api_model.pt"):
    # initialize neural net
    net = API_NN().to(device).to(torch.float)
    if (load):
        print(f"Loading from {loadPath}")
        net.load_state_dict(torch.load(loadPath))
    else:
        print("Initializing weights...")
        net.apply(init_weights)

    # inialize losses for each iteration
    losses = []

    
    num_moves = 6
    num_scrambles = 100

    for m in range(num_iter):
        # update number of moves per scramble
        num_moves = min(int((m+1)/num_iter * 40)+1, 20)

        # get n scrambled cubes
        X = np.zeros((num_scrambles * num_moves, env.statesize))
        weights = np.zeros((num_scrambles * num_moves))
        Y = []

        print(f"Sampling #{m+1}")

        for i in tqdm.tqdm(range(num_scrambles), desc=f"numMoves = {num_moves}"):
            # get a random scramble algorithm by resetting the environment
            _, alg = env.reset(n=num_moves)
            # get the list of moves by splitting the string
            # (we get rid of double moves to avoid confusion)
            moves = str(alg).replace("2", "").split()
            #print(moves)
            # reset cube to solved to start iterating through the scramble
            cube, _ = env.reset(n=0)
            for j,move in enumerate(moves):
                # move the cube to the desired state
                cube, _, _ = env.step(move)
                X[i*num_moves + j,:] = encode(cube)
                weights[i*num_moves + j] = 1.0/(j+1)
                #env.render()

                # values for each action taken from scramble state
                values = np.zeros((env.actionsize))

                # enumerate through entire action
                for k,a in enumerate(env.action_list):
                    next_cube, r, done = env.step(a)
                    #env.render()
                    next_state = encode(next_cube)

                    # get policy and value
                    value = torch.FloatTensor([0])
                    policy = None
                    if not done:
                        policy,value = net(next_state)
                    # print(f"POLICY {k}: {policy}")
                    # print(value)
                    # print(f"VALUE {env.action_list[k]}: {value.item() + r}")
                    values[k] = value.item() + r

                    # revert the cube back
                    env.step(env.inverse[a])

                # get target value and policy
                maxval = np.argmax(values)
                p = torch.zeros(env.actionsize)
                p[maxval] = 1
                #print(env.action_list[maxval])
                v = values[maxval]
                #print(values[maxval])
                #print(p)
                Y.append([p,v])
        
        # normalize weights
        weights = weights * weights.size / np.sum(weights)

        # uncomment to debug
        # print(X)
        # print(weights)
        # print(Y)

        # train NN and collect loss
        print(f"Training #{m+1}")
        endloss = TrainNN(net,X,Y, weights,EPOCHS=10)
        losses.append(endloss)

        # save the model after each training iteration
        torch.save(net.state_dict(), "api_model.pt")

        # save the losses array
        np.save("api_loss.npy", losses)
        #print(losses)
    return net

# Uncomment to debug
print(device)

# env = Cube()
# API(env, num_iter=10000, load=True)
    
