import time
import ray
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import Qnet
from settings import *

@ray.remote(num_cpus=1, num_gpus=1)
class Learner():
	def __init__(self, num_actions, shared_state, shared_memory, saved_model = False):
		self.num_actions = num_actions
		self.gamma = 0.98
		self.batch_size = 32
		self.alpha = 0.6
		self.beta = 0.4

		self.learning_rate = 1e-4
		self.shared_memory = shared_memory
		self.shared_state = shared_state
		self.prioritized_replay_eps = 1e-5

		self.min_replay_memory_size = 1 * 10 ** 4
		self.q = Qnet(84, 84, in_channels=4,
					  n_actions=self.num_actions).to(device)
		self.q_target = Qnet(84, 84, in_channels=4,
							 n_actions=self.num_actions).to(device)
		self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

		if saved_model:
			# Load pretrained model
			self.q.load_state_dict(torch.load(saved_model))
			self.q_target.load_state_dict(
				self.q.state_dict())  # Load Q into Q_target

		self.shared_state.put_Q_dict.remote(self.q.cpu().state_dict())
		self.q.to(device)

		self.num_q_updates = 0
		self.q_target_update_freq = 400


	def train(self, experience_batch):
		s,a,r,s_prime, gamma, qS_tpn, qS_t, key, weights, idxes = experience_batch
		#s,a,r,s_prime,done_mask,weights,idxes = experience_batch

		weights = torch.Tensor(weights).to(device)
		s = torch.as_tensor(s).to(device)
		a = torch.LongTensor(a).unsqueeze(1).to(device)
		r = torch.as_tensor(r).unsqueeze(1).to(device)
		s_prime = torch.as_tensor(s_prime).to(device)
		gamma = torch.Tensor(gamma).unsqueeze(1).to(device)
		#done_mask = torch.as_tensor(done_mask).to(device)

		# Q_out is the observed transitions given the current network
		q_out = self.q(s)
		# collect output from the chosen action dimension
		q_a = q_out.gather(1,a)

		# DDQN Update
		argmax_q = self.q(s_prime).argmax(1).unsqueeze(1)
		q_prime = self.q_target(s_prime).gather(1,argmax_q)
		target = r + self.gamma * q_prime# * done_mask

		TD_errors = q_a - target
		with torch.no_grad():
			new_priorities = np.abs(TD_errors.cpu()) + self.prioritized_replay_eps

		loss = ((TD_errors ** 2).view(-1) * weights).mean()
		return (idxes, new_priorities.numpy().squeeze()), loss


	def update_q(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.num_q_updates += 1
		if self.num_q_updates % self.q_target_update_freq == 0:
			self.q_target.load_state_dict(self.q.state_dict())
			print("Updating Q_target to q: ", self.num_q_updates)



	def learn(self, T):
		'''
		Sample a batch of experiences from shared buffer
		Compute loss and calculate new priorities
		Update parameters
		Send new parameters to parameter server
		Set new priorities
		Remove old experience from replay memory
		'''
		while ray.get(self.shared_memory.size.remote()) <= self.min_replay_memory_size:
			print("Shared Memory Size: ", ray.get(self.shared_memory.size.remote()))
			time.sleep(1)
		for t in range(1,T):
			experience_batch = ray.get(self.shared_memory.sample.remote(self.batch_size, self.beta))
			new_priorities, loss = self.train(experience_batch)
			self.update_q(loss)
			self.shared_state.put_Q_dict.remote(self.q.cpu().state_dict())
			self.q.to(device)
			self.shared_memory.update_priorities.remote(*new_priorities)
			#if t % self.shared_replay_update_freq == 0:
			#	self.shared_memory.remove_old_experience()
