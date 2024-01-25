import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.distributions import MultivariateNormal


from seller_policy_model import SellerRLAgent


class PPO:
    def __init__(self,agents,**hyperparameters):
        self._init_hyperparameters()
        self.agents = agents
        self.obs_dim = 39
        self.act_dim = 1
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.actor_dict = {f'agent{i+1}':SellerRLAgent(self.obs_dim, self.act_dim) for i in range(self.agents)}
        self.critic_dict = {f'agent{i+1}':SellerRLAgent(self.obs_dim, 1) for i in range(self.agents)}
        self.actor_optim_dict = {f'agent{i+1}':Adam(self.actor_dict[f'agent{i+1}'].parameters(), lr=self.lr) for i in range(self.agents)}
        self.critic_optim_dict = {f'agent{i+1}':Adam(self.critic_dict[f'agent{i+1}'].parameters(), lr=self.lr) for i in range(self.agents)}

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': {f'agent{i+1}':[] for i in range(self.agents)},     # losses of actor network in current iteration
            "avg_batch_rews": {f'agent{i+1}':[] for i in range(self.agents)} ,    # avg episodic returns in batch
            "avg_actor_losses": {f'agent{i+1}':[] for i in range(self.agents)}    # avg losses of actor network in current iteration
        }



    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048            # timesteps per batch
        self.max_timesteps_per_episode = 200      # timesteps per episode

        self.gamma = 0.99 #discount factor

        self.n_updates_per_iteration = 10

        self.lr = 3e-4

        self.clip = 0.2

        self.save_freq = 1

        self.seed = None

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

            # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")


    def get_action(self, obs):
      actions = []
      log_probs = []
      for i in range(self.agents):
        mean = self.actor_dict[f'agent{i+1}'](obs[i,:])
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        actions.append(action.detach().numpy())
        log_probs.append(log_prob.detach().numpy())

      # Return the sampled action and the log prob of that action
      return np.array(actions),np.array(log_probs)


    def compute_rtgs(self,batch_rews):

        batch_rtgs = []
        batch_shape = len(batch_rews[0])*len(batch_rews)
        # Iterate through each episode backwards to maintain same order in batch_rtgs

        for ep_rews in reversed(batch_rews):
            s  = []
            for i in range(self.agents):
              s+=[0.0]
            discounted_reward = np.array(s).reshape(batch_rews[0][0].shape) # The discounted reward so far
            ep_rtgs = []
            for rew in reversed(ep_rews):
              discounted_reward = rew + discounted_reward *self.gamma
              ep_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(ep_rtgs)


        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).reshape(batch_shape,self.agents)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts,ag):
      V = self.critic_dict[f'agent{ag+1}'](batch_obs).squeeze()
      mean = self.actor_dict[f'agent{ag+1}'](batch_obs)
      dist = MultivariateNormal(mean, self.cov_mat)
      log_probs = dist.log_prob(batch_acts)
      return V, log_probs

    def _log_summary(self,ag):

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in np.array(self.logger['batch_rews'])[:,:,ag,:]])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses'][f'agent{ag+1}']])
        self.logger['avg_batch_rews'][f'agent{ag+1}'].append(avg_ep_rews)
        self.logger['avg_actor_losses'][f'agent{ag+1}'].append(avg_actor_loss)
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Displaying the stats for the agent: {ag+1}", flush = True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        #self.logger['batch_lens'] = []
        #self.logger['batch_rews'] = []
        self.logger['actor_losses'][f'agent{ag+1}'] = []
