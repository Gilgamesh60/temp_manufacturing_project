import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torch.distributions import MultivariateNormal
import numpy as np
import numpy as np
import copy
import math
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")


def get_buyer_quantities(self,agent):
      '''
      Takes the current seller agent as input.
      Treats all the other agents in the network except the current seller agent as a buyer.
      For each buyer agent => we define spot demand quantitites and exchange demands for each buyer-seller pair.
      Then for each buyer, we find the spot and exchange demand qtys that maximize the buyer reward which takes into
      consideration the exchange prices of all the sellers.

      For eg. For 4-agent environment, if agent 2 is a buyer , agent 1,3 and 4 are considered as sellers and their exchange
      prices are cosidered in the buyer reward of agent 2.
      
      For the given seller, the function returns the spot and exchange demands of all the other buyer agents and also their 
      corresponding buyer rewards.
      '''
      buyer_spot_quantities = {}
      buyer_exchange_quantities = {}
      waste_exchange_quantities = {}
      for j in range(self.agents):
        buyer_spot_quantities[f'agent{j+1}'] = cp.Variable(nonneg = True)
        buyer_exchange_quantities[f'agent{j+1}'] = {}
        waste_exchange_quantities[f'agent{j+1}'] = {}
        for i in range(self.agents):
          buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'] =  cp.Variable(nonneg = True)
          waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'] =  cp.Variable(nonneg = True)
      
      buyer_spot_qtys = {}  
      buyer_exchange_qtys = {} 
      waste_exchange_qtys  = {} 
      buyer_reward_total = {}

      # this loop excludes the seller agent and calculates the buyer reward function for all the other buyers.
      for j in range(self.agents):
        if f'agent{j+1}' == agent:
          continue
        total_demand_by_agent = buyer_spot_quantities[f'agent{j+1}'] 
        for i in range(self.agents):
          if f'agent{i+1}' == f'agent{j+1}':
            continue
          total_demand_by_agent += buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']
          total_demand_by_agent += waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']

        function = self.utility(total_demand_by_agent) - self.complete_dict[f'data_dict_agent{j+1}']['spot_price'][self.t]*buyer_spot_quantities[f'agent{j+1}'] - self.transport_cost(buyer_spot_quantities[f'agent{j+1}'])
        for i in range(self.agents):
          if f'agent{i+1}' == f'agent{j+1}':
            continue
          function -= self.exchange_price_action_dict[f'agent{i+1}']*(buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']+waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'])
          function -= self.transport_cost(buyer_exchange_quantities[f'agent{j+1}'][f'agent{i+1}']+waste_exchange_quantities[f'agent{j+1}'][f'agent{i+1}'])
        
        objective = cp.Maximize(function)
        constraints = [total_demand_by_agent<=self.complete_dict[f'data_dict_agent{i+1}']['buyer_init_inventory']]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        buyer_reward_total[f'agent{j+1}'] = problem.value
        buyer_spot_qtys[f'agent{j+1}'] = buyer_spot_quantities[f'agent{j+1}'].value
        buyer_exchange_qtys[f'agent{j+1}']   = buyer_exchange_quantities[f'agent{j+1}'][agent].value
        waste_exchange_qtys[f'agent{j+1}'] = waste_exchange_quantities[f'agent{j+1}'][agent].value
      
      return buyer_spot_qtys,buyer_exchange_qtys,waste_exchange_qtys,buyer_reward_total
