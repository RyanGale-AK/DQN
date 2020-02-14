import gym
import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image

from models import Qnet, DuelingQnet
from settings import device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(obs):
    state = torch.ByteTensor(obs)
    return state.permute((2,0,1)).unsqueeze(0)

# record trained agent gameplay
def saveTrainedGameplay(env, target_bot):
    game = env.unwrapped.game
    env = gym.wrappers.Monitor(env, './videos/' + game, force=True)
    q = Qnet(84,84, in_channels = 4, n_actions = env.action_space.n).to(device)
    q.load_state_dict(torch.load(target_bot,map_location=device))
    q.eval()

    # Reset Environment for each game
    state = get_state(env.reset())
    episode_score = 0
    done = False
    epsilon = 0.0
    while not done:
        action = q(state.to(device)).max(1)[1].view(1,1)

        obs, reward, done, info = env.step(action)

        next_state = get_state(obs)

        state = next_state
        if done:
            break
    env.close()
