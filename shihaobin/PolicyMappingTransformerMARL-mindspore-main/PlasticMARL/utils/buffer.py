import numpy as np
from msadapter.pytorch import Tensor
from msadapter.pytorch.autograd import Variable

class ReplayBuffer(object):

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, h_dims):

        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.pre_acts = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.next_acts = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims): #(18,3),3
            self.obs_buffs.append(np.zeros((max_steps, odim[0],odim[1]), dtype=np.float32))
            self.pre_acts.append(np.zeros((max_steps, h_dims), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, odim[0],odim[1]), dtype=np.float32))
            self.next_acts.append(np.zeros((max_steps, h_dims), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations,pre_actions, actions, rewards, next_observations,next_actions, dones):

        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.pre_acts[agent_i] = np.roll(self.pre_acts[agent_i],
                                                 rollover,axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.next_acts[agent_i] = np.roll(self.pre_acts[agent_i],
                                                  rollover,axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[0])
            self.pre_acts[agent_i][self.curr_i:self.curr_i+nentries] = pre_actions[agent_i].detach().numpy()
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[0])
            self.next_acts[agent_i][self.curr_i:self.curr_i+nentries]=next_actions[agent_i].detach().numpy()
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]

        return ([cast(self.obs_buffs[i][inds]).transpose(0,1) for i in range(self.num_agents)],
                [cast(self.pre_acts[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]).transpose(0,1) for i in range(self.num_agents)],
                [cast(self.next_acts[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
