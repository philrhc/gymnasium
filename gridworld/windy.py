import subprocess
import gymnasium as gym
from collections import defaultdict
from gymnasium.core import ActType
import numpy as np

max_x = 9
max_y = 6

reward_x = 7
reward_y = 3

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


class HumanAgent:

    def act(self):
        return input()


class GridWorld(gym.Env):

    def __init__(self, render_mode):
        self.x = 0
        self.y = 3
        self.render_mode = render_mode

    def step(self, action: ActType):
        terminated = False
        reward = -1
        match action:
            case "0":
                if self.y < max_y:
                    self.y += 1
            case "1":
                if self.x < max_x:
                    self.x += 1
            case "2":
                if self.y > 0:
                    self.y -= 1
            case "3":
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


gridworld = GridWorld("human")
gridworld.render()
terminated = False
while not terminated:
    action = HumanAgent().act()
    _, _, terminated, _, _ = gridworld.step(action)
    subprocess.call("clear", shell=True)
    gridworld.render()
