import torch
import numpy as np
import random
import ray

from collections import namedtuple

from settings import *
from segment_tree import SumSegmentTree, MinSegmentTree


# @ray.remote(num_cpus=2, num_gpus=0)
# class Replay():
#    def __init__(self):
#        self.buffer_limit = 10 ** 6
#        self.alpha = 0.6
#    self.memory = PrioritizedReplayBuffer(
#        size=self.buffer_limit, alpha=self.alpha)
#    self.prioritized_replay_eps = 1e-5

"""
store previous sequence-action pairs to decorrelate temporal locality
transitions are composed of state, action, reward, next_state, and done_mask
"""


class ReplayBuffer():
    def __init__(self, size, transition_func=lambda *args: args):
        self.buffer = []
        self.maxsize = size
        self.next_idx = 0

        self.transition_func = transition_func

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def put(self, transition):
        transition = self.transition_func(*transition)
        if self.next_idx >= len(self.buffer):
            self.buffer.append(transition)
        else:
            self.buffer[self.next_idx] = transition
        self.next_idx = (self.next_idx + 1) % self.maxsize

    def _encode_sample(self, idxes):
        s_lst, a_lst, r_lst, next_state_lst, done_mask_lst = [], [], [], [], []

        for index in idxes:
            transition = self.buffer[index]
            s, a, r, next_state, done_mask = transition
            s_lst.append(np.array(s, copy=False))
            a_lst.append(np.array([a], copy=False))
            r_lst.append(np.array([r], copy=False))
            next_state_lst.append(np.array(next_state, copy=False))
            done_mask_lst.append(np.array([done_mask], copy=False))

        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(next_state_lst), np.array(done_mask_lst)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.buffer) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


'''
	The n-step replay buffer has two separate buffers. 
	The first is the running, cylic buffer that contains partial transitions:
		(S_t, A_t, R_t:t+n,gamma_t:t+n, q(S_t,*))
	and a batched n-step transition buffer with elements:
		(S_t, A_t, R_t:t+n,gamma_t:t+n, q(S_t,*), q(S_t+n,*), S_t+n)
	these are occasionally added to the global replay buffer after computing priorities inside the actors


'''

Transition = namedtuple('Transition', ['S', 'A', 'R', 'Gamma', 'q'])
N_Step_Transition = namedtuple('N_Step_Transition', [
                               'S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'])


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, actor_id, size):
        super(NStepReplayBuffer, self).__init__(size, Transition)
        assert size > 0
        self.nstep_buffer = []
        self.id = actor_id
        self.nstep_seq_num = 0

    def update_transitions(self, new_idx):
        R = self.buffer[new_idx].R
        Gamma = self.buffer[new_idx].Gamma
        # we need to update the R and gamma values for all other values than new_idx
        for i in [*range(new_idx-1, -1, -1), *range(self.size()-1, new_idx, -1)]:
            new_R = self.buffer[i].R + R*self.buffer[i].Gamma
            new_Gamma = self.buffer[i].Gamma * Gamma

            self.buffer[i] = self.transition_func(self.buffer[i].S, self.buffer[i].A,
                                                  new_R, new_Gamma, self.buffer[i].q)

    def construct_nstep_transition(self, start_idx, end_idx):
        key = str(self.id) + str(self.nstep_seq_num)
        self.nstep_seq_num += 1
        S_end = self.buffer[end_idx].S
        q_end = self.buffer[end_idx].q
        transition = N_Step_Transition(
            *self.buffer[start_idx], S_end, q_end, key)
        self.nstep_buffer.append(transition)

    def put(self, transition, done=False):
        # if self.size() < self.maxsize:
        idx = self.next_idx
        super(NStepReplayBuffer, self).put(transition)
        self.update_transitions(idx)

        if done:
            for n_step_start_idx in range(0, self.size()):
                if n_step_start_idx != idx:
                    self.construct_nstep_transition(
                        n_step_start_idx, end_idx=idx)
            # if we finish episode then clear the local buffer for new n-steps next game
            self.buffer.clear()
            self.next_idx = 0
        elif self.size() >= self.maxsize:
            n_step_start_idx = self.next_idx
            self.construct_nstep_transition(n_step_start_idx, end_idx=idx)

    def sample(self, batch_size):
        assert batch_size <= len(
            self.nstep_buffer), 'Requested n-step transitions is more than available'
        batch = self.nstep_buffer[: batch_size]
        del self.nstep_buffer[: batch_size]
        return batch

    def nstep_size(self):
        return len(self.nstep_buffer)

# Rank-based Prioritized Replay


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def put(self, transition):
        idx = self.next_idx
        super().put(transition)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


@ray.remote(num_cpus=1, num_gpus=0)
class ReplayMemory(ReplayBuffer):
    def __init__(self, size, alpha):
        super().__init__(size, N_Step_Transition)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def put(self, transitions, priorities):
        idxes = []
        for transition in transitions:
            idx = self.next_idx
            super().put(transition)
            idxes.append(idx)
        self.update_priorities(idxes, priorities)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res



    def _encode_sample(self, idxes):
        s_lst, a_lst, r_lst, next_state_lst, done_mask_lst = [], [], [], [], []
        
        n_step_transitions = N_Step_Transition(*zip(*[self.buffer[index] for index in idxes]))
        
        # 'S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'
        S_t = np.array(n_step_transitions.S_t)
        S_tpn = np.array(n_step_transitions.S_tpn)
        R_ttpB = np.array(n_step_transitions.R_ttpB)
        gamma_ttpB = np.array(n_step_transitions.Gamma_ttpB)
        qS_tpn = np.array(n_step_transitions.qS_tpn)
        A_t = np.array(n_step_transitions.A_t, dtype=np.int)
        qS_t = np.array(n_step_transitions.qS_t)
        key = np.array(n_step_transitions.key)
        
        return S_t, A_t, R_ttpB, S_tpn, gamma_ttpB, qS_tpn, qS_t, key

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def remove_old_experience(self):
        if self.size() > self.maxsize:
            num_excess = self.size() - self.maxsize

            # FIFO
            del self.buffer[: num_excess]