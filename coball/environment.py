import pkg_resources
import random
import torch
import numpy as np

from unityagents import UnityEnvironment
from unityagents.exception import UnityEnvironmentException


PLATFORM_PATHS = ['Tennis.app',
        'Tennis_Windows_x86/Tennis.exe',
        'Tennis_Windows_x86_64/Tennis.exe',
        'Tennis_Linux/Tennis.x86',
        'Tennis_Linux_NoVis/Tennis.x86',
        'Tennis_Linux/Tennis.x86_64',
        'Tennis_Linux_NoVis/Tennis.x86_64']

class CoBallEnv:
    """Continuous control environment.

    The environment accepts actions and provides states and rewards in response.
    """

    def __init__(self):
        for path in PLATFORM_PATHS:
            try:
                unity_resource = pkg_resources.resource_filename('coball', 'resources/' + path)
                self._env = UnityEnvironment(file_name=unity_resource)
                print("Environment loaded from " + path)
                break
            except UnityEnvironmentException as e:
                print("Attempted to load " + path + ":")
                print(e)
                print("")
                pass

        if not hasattr(self, '_env'):
            raise Exception("No unity environment found, setup the environment as described in the README.")

        # get the default brain
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        self._info = None
        self._scores = None
        self._score_history = ()

    def generate_episode(self, agent, max_steps=None, episodic=True, train_mode=False):
        """Create a generator for and episode driven by an actor.
        Args:
            actor: An actor that provides the next action for a given state.
            max_steps: Maximum number of steps (int) to take in the episode. If
                None, the episode is generated until a terminal state is reached.

        Returns:
            A generator providing a tuple of the current state, the action taken,
            the obtained reward, the next state and a flag whether the next
            state is terminal or not.
        """
        states = self.reset(train_mode=train_mode)
        print(states.shape)
        is_terminal = False
        count = 0

        while (max_steps is None or count < max_steps) and (not episodic or not is_terminal):
            actions = agent.act(states)
            rewards, next_states, is_terminals = self.step(actions)

            step_data = (states, actions, rewards, next_states, is_terminals)

            states = next_states
            is_terminal = np.any(is_terminals)
            count += 1
            if np.any(np.isnan(states)):
                raise ValueError("Nan from env")
            # print("--- step " + str(count) + " ---")

            yield step_data

        self._score_history += (self.get_score(), )
        self._scores = None

    def reset(self, train_mode=False):
        """Reset and initiate a new episode in the environment.

        Args:
            train_mode: Indicate if the environment should be initiated in
                training mode or not.

        Returns:
            The initial state of the episode (np.array).
        """
        # if self._info is not None and not np.any(self._info.local_done):
        #     raise Exception("Env is active, call terminate first")

        if self._scores is not None:
            self._score_history += (np.mean(self._scores))

        self._info = self._env.reset(train_mode=train_mode)[self._brain_name]
        self._scores = np.zeros(self.get_agent_size())

        return self._info.vector_observations

    def step(self, actions):
        """Execute an action on all instances.

        Args:
            action: An tensor of ints representing the actions each instance.

        Returns:
            A tuple containing the rewards (floats), the next states (np.array) and
            a booleans indicating if the next state is terminal or not.
        """
        if self._info is None:
            raise Exception("Env is not active, call reset first")

        if torch.is_tensor(actions):
            actions = actions.numpy()

        self._info = self._env.step(actions)[self._brain_name]
        next_states = self._info.vector_observations
        rewards = self._info.rewards
        is_terminals = self._info.local_done
        self._scores += np.array(rewards)

        return rewards, next_states, is_terminals

    def terminate(self):
        self._info = None
        if self._scores is not None:
            self._score_history += (self.get_score(), )
        self._scores = None

    def close(self):
        self.terminate()
        self._env.close()

    def get_score(self):
        """Return the cumulative average reward of the current episode."""
        return np.mean(self._scores) if self._scores is not None else self._score_history[-1]

    def get_score_history(self):
        """Return the cumulative average reward of all episodes."""
        return self._score_history

    def clear_score_history(self):
        """Clear the cumulative average reward of all episodes."""
        self._scores = None
        self._score_history = ()

    def get_agent_size(self):
        if self._info is None:
            raise ValueError("No agents are initialized")

        return len(self._info.agents)

    def get_action_size(self):
        return self._brain.vector_action_space_size

    def get_state_size(self):
        return 3 * self._brain.vector_observation_space_size


class CoBallAgent:
    """Agent based on a policy approximator."""

    def __init__(self, policy):
        """Initialize the agent.

        Args:
            pi: policy-function that is callable with n states and returns a
                (n, a)-dim array-like containing the value of each action.
        """
        self._policy = policy

    def act(self, states):
        """Select actions for the given states.

        Args:
            state: An array-like of states to choose the actions for.
        Returns:
            An array-like of floats representing the actions.
        """
        if not torch.is_tensor(states):
            try:
                states = torch.from_numpy(states, dtype=torch.float)
            except:
                states = torch.from_numpy(np.array(states, dtype=np.float))
        else:
            states = states.float()

        with torch.no_grad():
            return self._policy.sample(states)

    def get_policy(self):
        return self._policy

if __name__ == '__main__':
    # Run as > PYTHONPATH=".." python environment.py
    from pprint import pprint

    env = CoBallEnv()
    print("Environment specs:")
    pprint(env.get_action_size())
    pprint(env.get_state_size())

    print("Reset:")
    pprint(env.reset())

    dummy_actions = [[random.uniform(-1, 1)] * env.get_action_size()] * env.get_agent_size()

    print("Action 0:")
    pprint(env.step(dummy_actions))
    print("Action 1:")
    pprint(env.step(dummy_actions))
    print("Action 2:")
    pprint(env.step(dummy_actions))
    print("Action 3:")
    pprint(env.step(dummy_actions))

    env.terminate()

    print("\n\n\nRun episode:")
    rand_pi = lambda s: torch.rand(env.get_agent_size(), env.get_action_size())
    agent = CoBallAgent(rand_pi)
    episode = enumerate(env.generate_episode(agent, max_steps=5))
    for count, step_data in episode:
        print("\n\nCount:")
        pprint(count)
        print("Step data:")
        pprint(step_data)

    env.close()
