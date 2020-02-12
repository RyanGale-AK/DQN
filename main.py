import os, argparse
import gym

from wrappers import make_env
from helpers import saveTrainedGameplay
from DQN import DQN
from DDQN import DDQN


# example usage:
# dqn = DQN(env, save_location, start_episode = 1, saved_model = None)
# dqn.run(num_episodes = 100)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN Model on PongNoFrameskip-v4')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--loadModel', '-l', type=str, default=None)
    parser.add_argument('--start_episode', type=int, default=1)
    parser.add_argument('--saveDirectory', '-s', type=str, default=None)
    parser.add_argument('--saveVideo', type=str, default=None)
    parser.add_argument('--model', type=str, default='DQN')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    args = parser.parse_args()

    env = gym.make(args.env)
    env = make_env(env)
    default_save_location = 'checkpoints/%s/%s/' % (args.env, args.model)
    save_location = args.saveDirectory if args.saveDirectory else default_save_location

    if not os.path.exists(os.path.dirname(save_location)):
        try:
            os.makedirs(os.path.dirname(save_location))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    if args.saveVideo:
        saveTrainedGameplay(env, args.saveVideo)
    else:
        if args.model == "DQN":
            model = DQN(env, save_location, args.start_episode, args.saveDirectory)
        elif args.model == "DDQN":
            model = DDQN(env, save_location, args.start_episode, args.saveDirectory)
        model.run(args.episodes)
