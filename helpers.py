import gym
import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image

from settings import device


def get_screen(env):
    # convert to channel,h,w dimensions
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
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

# record trained agent gameplay
def saveTrainedGameplay(target_bot):
    env = gym.make('PongNoFrameskip-v4')
    env = gym.wrappers.Monitor(env, './videos/dqn_pong_video', force=True)
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w, in_channels = 3, n_actions = 4).to(device)
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