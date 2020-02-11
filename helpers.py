import gym
import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image

from models import Qnet
from settings import device



def get_state(obs):
    state = torch.Tensor(obs)
    return state.permute((2,0,1)).unsqueeze(0)

# record trained agent gameplay
def saveTrainedGameplay(target_bot):
    env = gym.make('PongNoFrameskip-v4')
    env = gym.wrappers.Monitor(env, './videos/dqn_pong_video', force=True)
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w, in_channels = 4, n_actions = 4).to(device)
    q.load_state_dict(torch.load('checkpoints/%s.pt' % target_bot))
    q.eval()
    
    # Reset Environment for each game
    state = get_state(env.reset()).to(device)
    episode_score = 0
    done = False
    epsilon = 0.0
    while not done:
        env.render()
        action = q.sample_action(state, epsilon)
        
        obs, reward, done, info = env.step(action)

        next_state = get_state(obs).to(device)        
        
        state = next_state
        if done:
            break
    env.close()