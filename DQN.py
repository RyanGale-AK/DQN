import gym
import argparse
import math
import os
import tensorwatch as tw

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
# from apex import amp # playing around with mixed-precision training

# Local Imports
from models import Qnet
from wrappers import make_env
from memory import ReplayBuffer
from helpers import saveTrainedGameplay, get_state
from settings import device


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

    #with amp.scale_loss(loss, optimizer) as scaled_loss: # playing around with mixed-precision training
    	#scaled_loss.backward()
    loss.backward()
    optimizer.step()



def main(num_episodes, episode_start = 1, saved_model = None, save_loc = 'checkpoints/tmp/'):
    watcher = tw.Watcher()

    # Model parameters 
    learning_rate = 1e-4 # 0.0001 matches paper
    gamma = 0.98
    buffer_limit = 10 ** 5 # paper uses 1M last frames, but this is expensive, so we try 10x less
    batch_size = 32

    # Epsilon Decay Parameters
    eps_start = 1
    eps_end = 0.01
    decay_factor = 10 ** 5
    
    epsilon_decay = lambda x: eps_end + (eps_start - eps_end) * \
        math.exp(-1. * x / decay_factor)

    env = gym.make('PongNoFrameskip-v4')
    env = make_env(env)
    h, w = 84, 84


    # Initialize the policy (q) network, target network, and experience replay buffer
    q = Qnet(h,w, in_channels = 4, n_actions = 4).to(device)
    q_target = Qnet(h,w, in_channels = 4, n_actions = 4).to(device)
    memory = ReplayBuffer(buffer_limit = buffer_limit)

    if saved_model:
        # Set epsilon decay function to eps_end
        epsilon_decay = lambda x: eps_end
        # Load pretrained model
        q.load_state_dict(torch.load(saved_model))

    # Load policy weights into target network
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    save_interval = 250
    print_interval = 1
    update_target_interval = 1000 # every 1000 frames
    score = 0.0
    

    #[q, q_target], optimizer = amp.initialize([q, q_target], optimizer, opt_level="O1") #playing around with mixed-precision training
    total_frames = 0
    best_episode_score = -100
    for episode in tqdm(range(episode_start,episode_start + num_episodes)):
        # anneal 100% to 1% over training
        epsilon = epsilon_decay(total_frames)

        # Reset Environment for each game
        state = get_state(env.reset())
        episode_score = 0
        done = False
        while not done:
            total_frames += 1
            action = q.sample_action(state.to(device), epsilon)
            
            obs, reward, done, info = env.step(action)

            next_state = get_state(obs)

            done_mask = 0.0 if done else 1.0

            memory.put((state,action,reward,next_state,done_mask))
            
            state = next_state
            
            score += reward
            episode_score += reward

            if memory.size() > 10000:
	            train(q, q_target, memory, optimizer, batch_size, gamma)
            if total_frames%update_target_interval == 0:
                q_target.load_state_dict(q.state_dict())
            if done:
                break
        
        if episode_score > best_episode_score:
            best_episode_score = episode_score
            torch.save(q_target.state_dict(), os.path.join(save_loc, 'best_target_bot.pt'))
            torch.save(q.state_dict(), os.path.join(save_loc, 'best_policy_bot.pt'))

        if episode%print_interval==0 and episode!=0:
            print("n_episode : {}, Total Frames : {}, Average Score : {:.1f}, Episode Score : {:.1f}, Best Score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                episode, total_frames, score/episode, episode_score, best_episode_score, memory.size(), epsilon*100))
        
        if episode%save_interval==0 and episode!=0:
            # save model weights 
            torch.save(q_target.state_dict(), os.path.join(save_loc, 'target_bot_%s.pt' % episode))
            torch.save(q.state_dict(), os.path.join(save_loc, 'policy_bot_%s.pt' % episode))

        # Microsoft Tensorwatch Watcher for Visualizing Training
        watcher.observe(
            episode = episode,
            episode_score = episode_score,
            total_score = score,
            buffer_size = memory.size(),
            epsilon = epsilon,
            frames = total_frames,
        )

    # save final model weights 
    torch.save(q_target.state_dict(), os.path.join(save_loc, 'target_bot_final.pt'))
    torch.save(q.state_dict(), os.path.join(save_loc, 'policy_bot_final.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN Model on PongNoFrameskip-v4')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--loadModel', '-l', type=str, default=None)
    parser.add_argument('--start_episode', type=int, default=1)
    parser.add_argument('--saveLoc', '-s', type=str, default=None)
    parser.add_argument('--saveVideo', type=str, default=None)
    args = parser.parse_args()
    
    botLocation = args.saveVideo
    if botLocation is not None:
        saveTrainedGameplay(botLocation)
    else:
        if args.saveLoc is None:
            main(args.episodes, episode_start = args.start_episode, saved_model = args.loadModel)
        else:
            main(args.episodes, episode_start = args.start_episode, saved_model = args.loadModel, saveLoc = args.saveLoc)
  
