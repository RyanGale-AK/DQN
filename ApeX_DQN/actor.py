import time
import ray
import numpy as np
import gym
import math

from collections import namedtuple

from models import Qnet
from replay import NStepReplayBuffer
from settings import *
from wrappers import make_env
from helpers import saveTrainedGameplay, get_state

#device = 'cpu'
Transition = namedtuple('Transition', ['S', 'A', 'R', 'Gamma', 'q'])
N_Step_Transition = namedtuple('N_Step_Transition', [
                               'S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'])

@ray.remote(num_cpus=1, num_gpus=0)
class Actor():
	def __init__(self, actor_id, env, shared_memory, shared_state, n, saved_model=None):
		self.id = actor_id
		self.initialize_environment(env)

		self.shared_memory = shared_memory
		self.shared_state = shared_state

		self.num_actions = self.env.action_space.n
		self.saved_model = saved_model

		self.alpha = 0.6
		self.beta = 0.4

		self.learning_rate = 1e-4
		self.gamma = 0.98
		self.buffer_limit = 10 ** 5
		self.training_frame_start = 10000 * 5
		self.batch_size = 32

		self.eps_start = 1
		self.eps_end = 0.01
		self.decay_factor = 10 ** 5

		self.local_buffer = NStepReplayBuffer(actor_id, size=n)
		self.n_step_transition_batch_size = 5
		self.prioritized_replay_eps = 1e-5

		if saved_model:
			self.epsilon_decay = lambda x: self.eps_end
		else:
			self.epsilon_decay = lambda x: self.eps_end + \
				(self.eps_start - self.eps_end) * \
				 math.exp(-1. * x / self.decay_factor)

		self.save_interval = 100000
		self.update_parameters_interval = 100

		self.device = device

		time.sleep(10)
		self.q = Qnet(84, 84, in_channels=4,
					  n_actions=self.num_actions)#.to(device)
		q_weights = ray.get(self.shared_state.get.remote())['q']
		self.q.load_state_dict(q_weights)

	def initialize_environment(self, env):
		self.env = make_env(gym.make(env))

	def compute_priorities(self, n_step_transitions):
		n_step_transitions = N_Step_Transition(*zip(*n_step_transitions))
		R_ttpB = np.array(n_step_transitions.R_ttpB)
		gamma_ttpB = np.array(n_step_transitions.Gamma_ttpB)
		qS_tpn = np.array(n_step_transitions.qS_tpn)
		A_t = np.array(n_step_transitions.A_t, dtype=np.int)
		qS_t = np.array(n_step_transitions.qS_t)

		nstep_td_target = R_ttpB + gamma_ttpB * np.max(qS_tpn, 1)

		nstep_td_error = nstep_td_target - \
			np.array([qS_t[i, A_t[i]] for i in range(A_t.shape[0])])
		#priorities = {
			#k: val for k in n_step_transitions.key for val in abs(nstep_td_error)}
		return [abs(i) for i in nstep_td_error]
	'''
	B: batch size of n-step transitions
	T: number of episodes to run actor for
	'''

	def run(self, T, B=5):
		gamma = 0.98
		# Load a pretrained model
		if self.saved_model:
			# Load pretrained model
			self.q.load_state_dict(torch.load(saved_model))

		env = self.env
		score = 0.0
		best_episode_score = 0
		total_frames = 0
		state = get_state(env.reset())  # Start first game
		for episode in range(1, T):
			# anneal 100% to 1% over training
			epsilon = self.epsilon_decay(total_frames)
			episode_score = 0
			done = False
			while not done:
				# Select action using current policy (depends on annealing epsilon)
				with torch.no_grad():
					s = torch.Tensor(state).unsqueeze(0)#.to(device)
					action = self.q.sample_action(s, epsilon)
					q = self.q(s).squeeze().numpy()
				# Apply the action in the environment
				obs, reward, done, _ = env.step(action)
				# Add data to the local buffer
				self.local_buffer.put(
					(state, action, reward, gamma, q), done=done)
				state = get_state(obs)

				score += reward
				episode_score += reward

				# Send a batch of size B to the experience replay memory every B frames
				if self.local_buffer.nstep_size() >= B:
					batch = self.local_buffer.sample(B)
					p = self.compute_priorities(batch)
					# async: add the batch to the replay buffer
					self.shared_memory.put.remote(batch,p)

				# Async call to get the newest parameters
				if total_frames % self.update_parameters_interval == 0:
					#print("Actor ", self.id, "Getting latest parameters from server.", total_frames)
					updated_parameters = ray.get(self.shared_state.get.remote())['q']
					self.q.load_state_dict(updated_parameters) # update the network parameters

				# Reset environment for the next game
				if done:
					best_episode_score = max(best_episode_score, episode_score)
					state = get_state(env.reset())
					out = " Actor_id: {}, n_episode : {}, Total Frames : {}, Average Score : {:.1f}, Episode Score : {:.1f}, Best Score : {:.1f}, eps : {:.1f}%".format(
						self.id, episode, total_frames, score/episode, episode_score, best_episode_score, epsilon*100)
					print(out)
				total_frames += 1
	def debug(self):
		return self.local_buffer