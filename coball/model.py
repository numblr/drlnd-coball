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


class PolicyOU():
    """Policy that provides action probabilities and samples actions accordingly."""

    def __init__(self, model, action_size,
        mu=0.,
        # theta=1.0, sigma=0.1, init_offset=0.3, init_var=0.01, dt=1e-3,
        # theta=0.5, sigma=0.2, init_offset=0.3, init_var=0.01, dt=1e-1,
        theta=0.8, sigma=0.1, init_offset=0.3, init_var=0.01, dt=1e-2,
        seed=2, cap=[-1.0, 1.0]):
        """Initialize parameters.
        """
        if sigma <= 0.0:
            raise ValueError("sigma must be positive: " + str(sigma))

        self._model = model
        self._noise = OUNoise(action_size, mu=mu, theta=theta, sigma=sigma,
                init_offset=init_offset, init_var=init_var, dt=dt)
        self.reset()

        self._cap_min = cap[0] if cap is not None else None
        self._cap_max = cap[1] if cap is not None else None

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self._noise.reset()

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

        distributions = self.distributions(states)

        log_probs = distributions.log_prob(actions).float()
        if self._cap_max is not None:
            boundary = torch.ge(actions, self._cap_max)
            if boundary.any():
                log_cdf = torch.log(1 - distributions.cdf(actions)).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)
        if self._cap_min is not None:
            boundary = torch.le(actions, self._cap_min)
            if boundary.any():
                log_cdf = torch.log(distributions.cdf(actions)).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)

        return torch.sum(log_probs, dim=2).unsqueeze(2)

    def distributions(self, states):
        means = self._model(states.to(device, dtype=torch.float))

        noise_mean, noise_std = self._noise.distribution()

        noise_mean = noise_mean.unsqueeze(0).unsqueeze(0)
        noise_std = noise_std.unsqueeze(0).unsqueeze(0)

        return dist.Normal(means + noise_mean, noise_std)

    def sample(self, states):
        """Sample actions for the given states according to the policy.

        Params
        ======
        states (tensor): dimensions (agents, steps, state_size)

        Returns
        ======
        actions (tensor): dimensions (agents, steps, action_size)
        """
        means = self._model(states.to(device, dtype=torch.float))

        samples = means + self._noise.sample()

        if self._cap_min is not None and self._cap_max is not None:
            samples = torch.clamp(samples, self._cap_min, self._cap_max)

        return samples


class Policy():
    """Policy that provides action probabilities and samples actions accordingly."""

    def __init__(self, model, sigma=0.05, sigma_explore=0.5, epsilon=0.1, cap=[-1.0, 1.0]):
        """Initialize parameters.
        Params
        ======
            model fn: state -> means: A model that maps states to means of the
                  action distributions
            sigma (float): variance of the action distributions
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
        dist_exploit, dist_explore = self.distributions(states)
        if self._epsilon == 0.0 or random.uniform(0, 1) > self._epsilon:
            sample = dist_exploit.sample()
        else:
            sample = dist_explore.sample()

        if self._cap_min is None or self._cap_max is None:
            return sample
        else:
            return torch.clamp(sample, self._cap_min, self._cap_max)


class PolicyPlain():
    """Policy that provides action probabilities and samples actions accordingly."""

    def __init__(self, model, sigma=1.0, sigma_explore=0.5, epsilon=0.1, cap=[-1.0, 1.0]):
        """Initialize parameters.
        Params
        ======
            model fn: state -> means: A model that maps states to means of the
                  action distributions
            sigma (float): variance of the action distributions
            cap (list): limits for the action values
        """
        if sigma <= 0.0:
            raise ValueError("sigma must be positive: " + str(sigma))

        self._model = model
        self._sigma = sigma
        self._cap_min = cap[0] if cap is not None else None
        self._cap_max = cap[1] if cap is not None else None

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

        distributions = self.distributions(states)

        log_probs = distributions.log_prob(actions).float()
        if self._cap_max is not None:
            boundary = torch.ge(actions, self._cap_max)
            if boundary.any():
                log_cdf = torch.log(1 - distributions.cdf(actions)).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)
        if self._cap_min is not None:
            boundary = torch.le(actions, self._cap_min)
            if boundary.any():
                log_cdf = torch.log(distributions.cdf(actions)).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)

        return torch.sum(log_probs, dim=2).unsqueeze(2)

    def distributions(self, states):
        means = self._model(states.to(device, dtype=torch.float))

        return dist.Normal(means, self._sigma)

    def sample(self, states):
        """Sample actions for the given states according to the policy.

        Params
        ======
        states (tensor): dimensions (agents, steps, state_size)

        Returns
        ======
        actions (tensor): dimensions (agents, steps, action_size)
        """
        sample = self.distributions(states).sample()

        if self._cap_min is None or self._cap_max is None:
            return sample
        else:
            return torch.clamp(sample, self._cap_min, self._cap_max)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=2, mu=0., theta=0.15, sigma=0.2,
            init_offset=0.3, init_var=0.3, dt=1e-2):
        """Initialize parameters and noise process."""
        self.mu = mu * torch.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.init_offset = init_offset
        self.init_var = init_var
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to initial value."""
        self._noise_state = self.init_offset * (torch.rand_like(self.mu) - 0.5).sign() \
            + self.init_var * torch.randn_like(self.mu)
        self.step = 0

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.step += 1
        if self.step % 1000 == 0:
            print("sampled " + str(self.step))

        x = self._noise_state
        dx = self.theta * (self.mu - x) * self.dt \
                + self.sigma * math.sqrt(self.dt) * torch.randn_like(self.mu)

        self._noise_state = x + dx

        return self._noise_state

    def distribution(self):
        mean = self.mu + (self._noise_state - self.mu) * math.exp(-self.theta*self.step*self.dt)
        std = math.sqrt(self.sigma**2/(2*self.theta) * (1 - math.exp(-2*self.theta*self.step*self.dt)))

        return mean, std*torch.ones_like(mean)


if __name__ == '__main__':
    noise = OUNoise(1, mu=0., theta=1.0, sigma=0.1, init_offset=0.3, init_var=0.01, dt=1e-3)
    noise = OUNoise(1, theta=0.8, sigma=0.1, init_offset=0.3, init_var=0.01, dt=1e-2, seed=2)
    noise_red = OUNoise(1, mu=0., theta=0.5, sigma=0.08, init_offset=0.3, init_var=0.01, dt=1e-1)

    samples = [ noise.sample() for i in range(int(1e+5)) ]
    red_samples = [ noise_red.sample() for i in range(int(1e+5)) ]

    import matplotlib
    from matplotlib import pyplot as plt

    # plt.plot(samples, linestyle="")
    plt.subplot(2, 1, 1)
    plt.plot(samples, c='b', linestyle="",marker=",")
    plt.subplot(2, 1, 2)
    plt.plot(red_samples, c='r', linestyle="",marker=",")
    plt.show()
