import numpy as np
import copy
import math
import cvxpy as cp
from synthetic_data_generator import synthetic_exo_data_generator


exogenous_cost_attr = ['spot_price', 'holding_cost', 'waste_disposal_cost', 'exchange_transport_cost', 'spot_transport_cost']

exogenous_qty_attr = ['recycled_qty', 'waste_qty', 'produced_qty','total_demand_qty']

seller_nonexo_state_attributes = ['current_inventory', 'action']

seller_action_attributes = ['exchange_price_action']

buyer_action_attributes = ['buyer_spot_qty', 'buyer_exchange_qty']


class Manufacturing_Simulator:

    def __init__(self, agents = 2,PPO = None, decay_factor = 0.1, coefs = [300,400,10,20], T = 2048, history_length = 32,max_timesteps_per_episode= 200):

        self.agents = agents
        self.PPO = PPO
        self.coefs = coefs
        self.T = T
        self.lead_time = history_length
        self.max_timesteps_per_episode =max_timesteps_per_episode
        self.days_num = 0
        self.obs_dim = 38
        self.act_dim = 1
        self.current_inventory_dict = {}
        self.complete_dict = {}
        self.obs_dict = {}
        self.current_obs = None
        self.exchange_price_action_dict = {}
        self.decay_factor = decay_factor

    def step(self,action):
      action = list(action)
      for i in range(len(action)):
        self.exchange_price_action_dict['agent{}'.format(i+1)] = action[i]

      total_reward = []
      buyer_reward = []
      seller_reward = []
      k = []
      # next_obs, inventory and reward for each agent
      for i in range(self.agents):
        #buyer decision calculation using get_buyer_quantities which optimizes the objective fn using cvxpy
        spot_demand_qty_agent, exchange_demand_qty_agent ,waste_exchange_qty_agent,buyer_reward_agent = self.get_buyer_quantities(f'agent{i+1}',self.complete_dict[f'data_dict_agent{i+1}'])

        prev_inventory_qty = self.current_inventory_dict[f'agent{i+1}']

        # waste qty calculation : Subtracting the decayed part and waste_exchange_qty from waste_qty[t] to get waste_qty[t+1]. We put max to ensure that waste_qty>=0.
        self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t+1] = max(0,self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t]
                                                                                 - waste_exchange_qty_agent
                                                                                 -self.decay_factor*self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t])

        # current inventory calculation step
        self.current_inventory_dict[f'agent{i+1}'] = max(0,prev_inventory_qty
                                                         + self.complete_dict['data_dict_agent{}'.format(i+1)]['produced_qty'][self.t]
                                                         - exchange_demand_qty_agent 
                                                         - min(waste_exchange_qty_agent,self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t]) 
                                                         - self.complete_dict['data_dict_agent{}'.format(i+1)]['waste_qty'][self.t])
        # next_obs calculation step
        total_demand_agent = spot_demand_qty_agent + exchange_demand_qty_agent + waste_exchange_qty_agent
        next_obs_agent = np.concatenate(
          (self.obs_dict['current_obs_agent{}'.format(i+1)][1:self.lead_time],np.array([total_demand_agent]),np.array([self.current_inventory_dict[f'agent{i+1}']]),np.array([self.complete_dict['data_dict_agent{}'.format(i+1)]['spot_price'][self.t+1]]),np.array([self.complete_dict['data_dict_agent{}'.format(i+1)]['waste_qty'][self.t+1]]),np.array([self.complete_dict['data_dict_agent{}'.format(i+1)]['produced_qty'][self.t+1]]),np.array([self.complete_dict['data_dict_agent{}'.format(i+1)]['holding_cost'][self.t+1]]),np.array([self.complete_dict['data_dict_agent{}'.format(i+1)]['seller_init_inventory']]))
      )
        k+=[next_obs_agent]

        # seller rewards , buyer rewards and total rewards by the agent.
        seller_reward_agent = self.exchange_price_action_dict['agent{}'.format(i+1)]*min(self.current_inventory_dict[f'agent{i+1}'],exchange_demand_qty_agent+waste_exchange_qty_agent) + self.transport_cost(exchange_demand_qty_agent+waste_exchange_qty_agent) - self.complete_dict['data_dict_agent{}'.format(i+1)]['waste_disposal_cost'][self.t] - self.complete_dict['data_dict_agent{}'.format(i+1)]['holding_cost'][self.t]
        total_reward_agent = buyer_reward_agent + seller_reward_agent
        total_reward.append(total_reward_agent)
        seller_reward.append(seller_reward_agent)
        buyer_reward.append(buyer_reward_agent)

      next_obs = np.array(k)
      self.t+=1
      done = False
      if self.t > self.max_timesteps_per_episode-1 :
        done = True

      self.current_obs = next_obs

      return next_obs , np.array(total_reward)  , done, {'buyer_reward':np.array(buyer_reward), 'seller_reward':np.array(seller_reward)}


    def reset(self):
      # reset step. returns the obs_dict which contains the basic state for all agents.
      self.t = 0
      for i in range(self.agents):
        self.complete_dict['data_dict_agent{}'.format(i+1)] =  synthetic_exo_data_generator(total_timesteps= self.T+self.lead_time)
        self.current_inventory_dict['agent{}'.format(i+1)] = 0
        self.obs_dict['current_obs_agent{}'.format(i+1)] =  np.array([0 for i in range(self.lead_time)]+[self.current_inventory_dict['agent{}'.format(i+1)]]+[self.complete_dict['data_dict_agent{}'.format(i+1)]['spot_price'][self.t]]+
                                        [self.complete_dict['data_dict_agent{}'.format(i+1)]['waste_qty'][self.t]]+[self.complete_dict['data_dict_agent{}'.format(i+1)]['produced_qty'][self.t]]+
                                         [self.complete_dict['data_dict_agent{}'.format(i+1)]['holding_cost'][self.t]]+[self.complete_dict['data_dict_agent{}'.format(i+1)]['seller_init_inventory']])

      s = []
      for i in range(self.agents):
        s+=[self.obs_dict['current_obs_agent{}'.format(i+1)]]

      self.current_obs =  np.array(s)
      return self.current_obs

    def utility(self,demand):
      return self.coefs[0]*demand + self.coefs[1]

    def waste_utility(self,waste_demand):
      return cp.power(waste_demand,0.50)


    def transport_cost(self,demand):
      return self.coefs[2]*demand + self.coefs[3]

    def waste_cost(waste_qty):
      pass

    def get_buyer_quantities(self,agent,dt):
        # function to return the spot and exchange demands by maxmizing the buyer objective function mentioned in the draft.
        
        buyer_spot_qty = cp.Variable(nonneg=True)
        buyer_exchange_qty = cp.Variable(nonneg=True)
        waste_exchange_qty = cp.Variable(nonneg=True)

        function = self.utility(buyer_spot_qty+buyer_exchange_qty)- dt['spot_price'][self.t]*buyer_spot_qty- self.transport_cost(buyer_spot_qty)
        function += self.waste_utility(waste_exchange_qty)
        function-= self.exchange_price_action_dict[agent]*buyer_exchange_qty
        function-= self.exchange_price_action_dict[agent]*waste_exchange_qty

        objective = cp.Maximize(function)
        total_demand = buyer_spot_qty+buyer_exchange_qty+waste_exchange_qty
        constraints = [total_demand<=dt['buyer_init_inventory']]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return buyer_spot_qty.value,buyer_exchange_qty.value,waste_exchange_qty.value,problem.value


    def rollout(self):
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Episodic data. Keeps track of wards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch+

        while t < self.PPO.timesteps_per_batch:
            ep_rews = []
            obs = self.reset()
            done = False

            for ep_t in range(self.PPO.max_timesteps_per_episode):
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.PPO.get_action(obs)
                obs, rew, done, _ = self.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

                # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)


        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        batch_rtgs = self.PPO.compute_rtgs(batch_rews)

        self.PPO.logger['batch_rews'] = batch_rews
        self.PPO.logger['batch_lens'] = batch_lens

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs , batch_lens


    def learn(self, total_timesteps):

        print(f"Learning... Running {self.PPO.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.PPO.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")


        t_so_far = 0 # Timesteps simulated so far

        i_so_far = 0


        while t_so_far < total_timesteps:

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            self.PPO.logger['t_so_far'] = t_so_far
            self.PPO.logger['i_so_far'] = i_so_far


            for ag in range(self.agents):
              V, _ = self.PPO.evaluate(batch_obs[:,ag,:], batch_acts[:,ag,:],ag)
              A_k = batch_rtgs[:,ag] - V.detach()
              A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

              for _ in range(self.PPO.n_updates_per_iteration):
                V, curr_log_probs = self.PPO.evaluate(batch_obs[:,ag,:], batch_acts[:,ag,:],ag)
                ratios = torch.exp(curr_log_probs - batch_log_probs[:,ag])
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.PPO.clip, 1 + self.PPO.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs[:,ag])
                self.PPO.actor_optim_dict[f'agent{ag+1}'].zero_grad()
                actor_loss.backward(retain_graph=True)
                self.PPO.actor_optim_dict[f'agent{ag+1}'].step()
                self.PPO.critic_optim_dict[f'agent{ag+1}'].zero_grad()
                critic_loss.backward()
                self.PPO.critic_optim_dict[f'agent{ag+1}'].step()
                self.PPO.logger['actor_losses'][f'agent{ag+1}'].append(actor_loss.detach())
              self.PPO._log_summary(ag)
              if i_so_far % self.PPO.save_freq == 0:
                torch.save(self.PPO.actor_dict[f'agent{ag+1}'].state_dict(), f'/content/ppo_actor_agent{ag+1}_{i_so_far}.pth')
                torch.save(self.PPO.critic_dict[f'agent{ag+1}'].state_dict(), f'/content/ppo_critic_agent{ag+1}_{i_so_far}.pth')
            self.PPO.logger['batch_rews'] = []
            self.PPO.logger['batch_lens'] = []
