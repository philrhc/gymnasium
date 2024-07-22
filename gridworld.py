from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from tqdm import tqdm

from agents.monte_carlo import MonteCarloAgent

max_x = 9
max_y = 6

reward_x = 7
reward_y = 3

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


class HumanAgent:
    def act(self):
        return input()


class GridWorld(gym.Env):

    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(4)
        self.x = 0
        self.y = 3
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return (self.x, self.y), None

    def step(self, action: ActType):
        terminated = False
        reward = -1
        match action:
            case 0:
                if self.y < max_y:
                    self.y += 1
            case 1:
                if self.x < max_x:
                    self.x += 1
            case 2:
                if self.y > 0:
                    self.y -= 1
            case 3:
                if self.x > 0:
                    self.x -= 1
        if self.y < max_y:
            self.y += wind[self.x]
            if self.y > max_y:
                self.y = max_y
        if self.x == reward_x and self.y == reward_y:
            terminated = True
            reward = 0
        return (self.x, self.y), reward, terminated, False, {}

    def render(self):
        sb = ""
        for i in reversed(range(0, max_y + 1)):
            for j in range(0, max_x + 1):
                if i == self.y and j == self.x:
                    sb += "O"
                    continue
                if i == reward_y and j == reward_x:
                    sb += "G"
                    continue
                sb += "_"
            sb += "\n"
        print(sb)
        print("Next move", end=": ")


env = GridWorld()
n_episodes = 10_000
learning_rate = 0.01
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = MonteCarloAgent(
    env.action_space,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
if agent.uses_training_error():
    training_error_moving_average = (
            np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()
exit()
