import torch
import ray
import json

from replay import ReplayBuffer, PrioritizedReplayBuffer
from actor import Actor
from learner import Learner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a DQN Model on PongNoFrameskip-v4')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--loadModel', '-l', type=str, default=None)
    parser.add_argument('--start_episode', type=int, default=1)
    parser.add_argument('--saveDirectory', '-s', type=str, default=None)
    parser.add_argument('--saveVideo', type=str, default=None)
    parser.add_argument('--model', type=str, default='DQN')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--prioritized', '-P', action='store_true')
    args = parser.parse_args()
    env = args.env
    default_save_location = 'checkpoints/%s/%s' % (env, args.model)
    if args.prioritized:
        default_save_location += '_prioritized'
    default_save_location += '/'
    save_location = args.saveDirectory if args.saveDirectory else default_save_location

    if args.saveVideo:
        saveTrainedGameplay(env, args.saveVideo)
    else:
        alg_args = tuple((env, save_location, args.start_episode,
                          args.loadModel, args.prioritized))
        if args.model == "DQN":
            model = DQN(*alg_args)
        elif args.model == "DDQN":
            model = DDQN(*alg_args)
        elif args.model == "DuelingDDQN":
            model = DuelingDDQN(*alg_args)
        model.run(args.episodes)


@ray.remote(num_cpus=1, num_gpus=0)
class ParameterServer():
    def __init__(self):
        self.parameters = dict()
        self.parameters['q'] = dict()

    def get(self):
        return self.parameters

    def put_Q_dict(self, new_q_dict):
        self.parameters['q'] = new_q_dict


if __name__ == "__main__":
    num_actors = 1
    B = 50
    periodic_update_frequency = 400
    shared_experience_replay_size = 2 * 10 * 6

    # PrioritizedReplayBuffer(shared_experience_replay_size, alpha = 0.6, beta = 0.4)
    shared_memory = ReplayMemory(shared_experience_replay_size, alpha=0.6)
    shared_state = ParameterServer.remote()

    learner = Learner.remote(
        num_actions, shared_memory=shared_memory, shared_state=shared_state)

    actors = [Actor.remote(env, shared_memory, shared_state)
              for _ in range(num_actors)]
