import torch
import numpy as np
import random

from settings import *
from segment_tree import SumSegmentTree, MinSegmentTree

"""
store previous sequence-action pairs to decorrelate temporal locality
transitions are composed of state, action, reward, next_state, and done_mask
"""
class ReplayBuffer():
    def __init__(self, size):
        self.buffer = [] #collections.deque(maxlen=size)
        self.maxsize = size
        self.next_idx = 0
    
    def __len__(self):
        return len(self.buffer)

    def put(self, transition):
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
            s_lst.append(np.array(s, copy = False))
            a_lst.append(np.array([a], copy = False))
            r_lst.append(np.array([r], copy = False))
            next_state_lst.append(np.array(next_state, copy = False))
            done_mask_lst.append(np.array([done_mask], copy = False))
            
        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(next_state_lst), np.array(done_mask_lst)        


    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

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