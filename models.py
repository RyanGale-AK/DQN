import random
import torch.nn as nn
import torch.nn.functional as F

from settings import device

class Qnet(nn.Module):
    def __init__(self, h, w, in_channels = 3, n_actions = 4):
        super(Qnet, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 4, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        self.linear_input_size = convw * convh * 64
        self.head = nn.Sequential(
            nn.Linear(self.linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    # action is either random or max probability estimated by Qnet
    def sample_action(self, obs, epsilon):
        out = self.forward(obs) # don't need if random action 
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            return out.argmax().item()


class DuelingQnet(Qnet):
    def __init__(self, h, w, in_channels, n_actions):
        super(DuelingQnet, self).__init__(h, w, in_channels, n_actions)
        self.n_actions = n_actions
        self.convolutions = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(),
            self.conv2, self.bn2, nn.ReLU(),
            self.conv3, self.bn3, nn.ReLU())

        self.value = nn.Sequential(
            nn.Linear(self.linear_input_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage = self.head

    def forward(self, x):
        x = x.float()
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        V = self.value(x)
        A = self.advantage(x)
        # Q = V + A - 1/|A| (A_1 + A_2 + ... + A_n)
        return V + A - A.mean(dim=1)[:, None]
