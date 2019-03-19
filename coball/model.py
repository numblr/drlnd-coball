from pprint import pprint

import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOModel(nn.Module):
    """Base module for Actor and Critic models."""

    def __init__(self, state_size, output_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            output (int): Dimension of the output
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PPOModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)


class Actor(PPOModel):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=2, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__(state_size, action_size, seed, fc1_units, fc2_units)
        self._reset_parameters()

    def _reset_parameters(self):
        self.fc1.bias.data.normal_(0.0, 3e-3)
        self.fc1.weight.data.uniform_(*self._init_limits(self.fc1))
        self.fc2.bias.data.normal_(0.0, 3e-3)
        self.fc2.weight.data.uniform_(*self._init_limits(self.fc2))
        self.fc3.bias.data.normal_(0.0, 3e-3)
        self.fc3.weight.data.uniform_(0.0, 3e-3)

    def _init_limits(self, layer):
        input_size = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(input_size)

        return (-lim, lim)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> action means."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return F.tanh(self.fc3(x))


class Critic(PPOModel):
    """Critic (Value) Model."""

    def __init__(self, state_size, seed=2, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__(state_size, 1, seed, fc1_units, fc2_units)
        self._reset_parameters()

    def _reset_parameters(self):
        self.fc1.weight.data.uniform_(*self._init_limits(self.fc1))
        self.fc2.weight.data.uniform_(*self._init_limits(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def _init_limits(self, layer):
        input_size = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(input_size)

        return (-lim, lim)

    def forward(self, state):
        """Build an critic (value function) network that maps states -> values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class Policy():
    """Policy that provides action probabilities and samples actions accordingly.

    The policy consists of a combination of two normal distributions, one with
    high variance to drive exploration and one with low variance to drive
    exploitation. The mean of both distributions is equal and provided by the
    underlying model.
    """

    def __init__(self, model, sigma=0.01, sigma_explore=0.5, epsilon=0.1, cap=[-1.0, 1.0], replay=False):
        """Initialize parameters.
        Params
        ======
            model fn: state -> means: A model that maps states to means of the
                  action distributions
            sigma (float): variance of the action distribution for exploitation
            sigma_explore (float): variance of the action distribution exploration
            epsilon: weight of the exploration distribution
            cap (list): limits for the action values
        """
        if sigma <= 0.0:
            raise ValueError("sigma must be positive: " + str(sigma))

        self._model = model
        self._sigma = sigma
        self._sigma_explore = sigma_explore
        self._epsilon = epsilon
        self._cap_min = cap[0] if cap is not None else None
        self._cap_max = cap[1] if cap is not None else None
        self._replay = replay

    def log_prob(self, states, actions):
        """Log-probabilities of the given actions for the given states.

        Params
        ======
        states (tensor): dimensions (agents, steps, state_size)
        actions (tensor): dimensions (agents, steps, action_size)

        Returns
        ======
        log probabilities (tensor): dimensions (agents, steps, 1)
        """
        if not states.dim() == actions.dim() == 3:
            raise ValueError("dimensions are too small, unsqueeze: " + str(states.size()) + "-" + str(actions.size()))

        if not states.size()[:2] == actions.size()[:2]:
            raise ValueError("dimenions don't match: " + str(states.size()) + "-" + str(actions.size()))

        states = states.to(device, dtype=torch.float)
        actions = actions.to(device, dtype=torch.float)

        dist_exploit, dist_explore = self.distributions(states)

        prob_exploit = torch.exp(dist_exploit.log_prob(actions).float() + math.log(1 - self._epsilon))
        prob_explore = torch.exp(dist_explore.log_prob(actions).float() + math.log(self._epsilon))
        log_probs = torch.log(prob_exploit + prob_explore)

        if self._cap_max is not None:
            boundary = torch.ge(actions, self._cap_max)
            if boundary.any():
                cdf = (1 - self._epsilon) * (1 - dist_exploit.cdf(actions).float()) \
                    + self._epsilon * (1 - dist_explore.cdf(actions).float())
                log_cdf = torch.log(cdf).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)
        if self._cap_min is not None:
            boundary = torch.le(actions, self._cap_min)
            if boundary.any():
                cdf = (1 - self._epsilon) * dist_exploit.cdf(actions).float() \
                        + self._epsilon * 1 - dist_explore.cdf(actions).float()
                log_cdf = torch.log(cdf).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)

        return torch.sum(log_probs, dim=2).unsqueeze(2)

    def distributions(self, states):
        means = self._model(states.to(device, dtype=torch.float))

        return (dist.Normal(means, self._sigma), dist.Normal(means, self._sigma_explore))

    def sample(self, states):
        """Sample actions for the given states according to the policy.

        Params
        ======
        states (tensor): dimensions (agents, steps, state_size)

        Returns
        ======
        actions (tensor): dimensions (agents, steps, action_size)
        """
        if self._replay:
            return self._model(states.to(device, dtype=torch.float))

        dist_exploit, dist_explore = self.distributions(states)
        if self._epsilon == 0.0 or random.uniform(0, 1) > self._epsilon:
            sample = dist_exploit.sample()
        else:
            sample = dist_explore.sample()

        if self._cap_min is None or self._cap_max is None:
            return sample
        else:
            return torch.clamp(sample, self._cap_min, self._cap_max)
