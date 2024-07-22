from __future__ import annotations
from collections import defaultdict
import numpy as np


class BlackjackAgent:
    def __init__(
            self,
            action_space,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,

    ):
        self.action_space = action_space
        self.q_values = defaultdict(lambda: np.zeros(action_space.n))
        self.num_q_observed = defaultdict(lambda: np.zeros(action_space.n))
        self.episode_observations = []
        self.episode_rewards = []
        self.episode_actions = []

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        self.episode_observations.append(obs)
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        if not terminated:
            return

        G = 0
        W = 1
        for index in reversed(range(len(self.episode_observations))):
            each_observation = self.episode_observations[index]
            each_action = self.episode_actions[index]

            G = self.discount_factor * G + self.episode_rewards[index]
            self.num_q_observed[each_observation][each_action] = self.num_q_observed[each_observation][each_action] + W
            self.q_values[each_observation][each_action] = self.q_values[each_observation][each_action] + (
                    W / self.num_q_observed[each_observation][each_action] * (
                    G - self.q_values[each_observation][each_action]))

        self.prepare_for_next_episode()

    def prepare_for_next_episode(self):
        self.episode_observations = []
        self.episode_rewards = []
        self.episode_actions = []

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def uses_training_error(self):
        return False
