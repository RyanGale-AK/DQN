import gym
import math
import random
import cv2
import numpy as np
import collections
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from wrappers import make_env

from tqdm import tqdm
# from apex import amp # playing around with mixed-precision training
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env):
    # convert to channel,h,w dimensions
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # erase background
    screen[screen == 72] = 0 
    screen[screen == 74] = 0 
    screen[screen == 144] = 0 
    screen[screen != 0] = 213
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    screen = torch.from_numpy(screen)
    
    # convert to batch,channel,h,w dimensions
    return resize(screen).unsqueeze(0).to(device)


"""
store previous sequence-action pairs to decorrelate temporal locality
transitions are composed of state, action, reward, next_state, and done_mask
"""
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            
        return  torch.cat(s_lst).to(device), \
                torch.tensor(a_lst).to(device), \
                torch.tensor(r_lst).to(device), \
                torch.cat(s_prime_lst).to(device), \
                torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, h, w, n_actions):
        super(Qnet, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
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
        linear_input_size = convw * convh * 64
        self.head = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def forward(self, x):
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


def train(q, q_target, memory, optimizer, batch_size, gamma):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    
    q_out = q(s)
    # collect output from the chosen action dimension
    q_a = q_out.gather(1,a) 
    
    # most reward we get in next state s_prime
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    # how much is our policy different from the true target 
    loss = F.smooth_l1_loss(q_a, target)
    
    optimizer.zero_grad()

    #with amp.scale_loss(loss, optimizer) as scaled_loss:
    	#scaled_loss.backward()
    loss.backward()
    optimizer.step()



def main(num_episodes, saved_model = None):
    learning_rate = 0.0001
    gamma = 0.98
    buffer_limit = 100000
    batch_size = 32

    env = gym.make('PongNoFrameskip-v4')
    env = make_env(env)
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w,4).to(device)

    if saved_model:
        print("Loading Model: ", saved_model)
        q.load_state_dict(torch.load('checkpoints/%s.pt' % saved_model))

    q_target = Qnet(h,w,4).to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit = buffer_limit)
    
    save_interval = 250
    print_interval = 1
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    #[q, q_target], optimizer = amp.initialize([q, q_target], optimizer, opt_level="O1") #playing around with mixed-precision training

    best_episode_score = -100
    for episode in tqdm(range(1,num_episodes+1)):
        # anneal 8% to 1% over training
        epsilon = max(0.01, 0.08 - 0.01*(episode/200))
        env.reset()
        current_s = get_screen(env)
        done = False
        last_s = get_screen(env)
        current_s = get_screen(env)
        s = last_s - current_s
        episode_score = 0
        while not done:
            a = q.sample_action(s, epsilon)
            # first variable would be s_prime but we have get_screen
            _, r, done, info = env.step(a)
            last_s = current_s
            current_s = get_screen(env)
            s_prime = last_s - current_s
            
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.,s_prime,done_mask))
            s = s_prime
            
            score += r
            episode_score += r
            if memory.size() > 5000:
	            train(q, q_target, memory, optimizer, batch_size, gamma)
            if done:
                break
        
        if episode_score > best_episode_score:
            best_episode_score = episode_score
            torch.save(q_target.state_dict(), 'checkpoints/4actions/best_target_bot.pt')
            torch.save(q.state_dict(), 'checkpoints/4actions/best_policy_bot.pt')

        if episode%print_interval==0 and episode!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, Average Score : {:.1f}, Episode Score : {:.1f}, Best Score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                episode, score/episode, episode_score, best_episode_score, memory.size(), epsilon*100))
            
        if episode%save_interval==0 and episode!=0:
            # save model weights 
            torch.save(q_target.state_dict(), 'checkpoints/4actions/target_bot_%s.pt' % episode)
            torch.save(q.state_dict(), 'checkpoints/4actions/policy_bot_%s.pt' % episode)
    # save final model weights 
    torch.save(q_target.state_dict(), 'checkpoints/4actions/target_bot_final.pt')
    torch.save(q.state_dict(), 'checkpoints/4actions/policy_bot_final.pt')

# record trained agent gameplay
def getTrainedGameplay(target_bot):
    env = gym.make('Pong-v0')
    env = gym.wrappers.Monitor(env, './videos/dqn_pong_video', force=True)
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w,4).to(device)
    q.load_state_dict(torch.load('checkpoints/%s.pt' % target_bot))
    q.eval()
    env.reset()
    current_s = get_screen(env)
    done = False
    last_s = get_screen(env)
    current_s = get_screen(env)
    s = last_s - current_s
    epsilon = 0.0
    while not done:
        a = q.sample_action(s, epsilon)
        
        # use environment's frame instead of preprocessed get_screen(env)
        _, _, done, info = env.step(a)
        last_s = current_s
        current_s = get_screen(env)
        s_prime = last_s - current_s

        done_mask = 0.0 if done else 1.0
        s = s_prime
        if done:
            break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN Model on Pong-v0')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--saveVideo', type=str, default=None)
    args = parser.parse_args()
    
    botLocation = args.saveVideo
    if botLocation:
        getTrainedGameplay(botLocation)
    else:
        main(args.episodes, saved_model = "4actions/target_bot_4000")
