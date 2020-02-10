import gym
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
# from apex import amp # playing around with mixed-precision training

# Local Imports
from models import Qnet
from wrappers import make_env
from memory import ReplayBuffer
from helpers import saveTrainedGameplay, get_screen
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



def main(num_episodes, saved_model = None, save_models = True):
    learning_rate = 0.0001
    gamma = 0.98
    buffer_limit = 100000
    batch_size = 32

    env = gym.make('PongNoFrameskip-v4')
    env = make_env(env)
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w, in_channels = 3, n_actions = 4).to(device)

    if saved_model:
        print("Loading Model: ", saved_model)
        q.load_state_dict(torch.load('checkpoints/%s.pt' % saved_model))

    q_target = Qnet(h,w, in_channels = 3, n_actions = 4).to(device)
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN Model on PongNoFrameskip-v4')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--saveVideo', type=str, default=None)
    args = parser.parse_args()
    
    botLocation = args.saveVideo
    if botLocation:
        saveTrainedGameplay(botLocation)
    else:
        main(args.episodes, saved_model = "4actions/target_bot_500", save_models = True)
