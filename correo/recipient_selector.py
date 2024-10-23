import numpy as np
import random


class LeastLatencySelector:
    def __init__(self, recipients, epsilon=0.1):
        self.epsilon = epsilon  # exploration rate
        self.n = recipients  # number of recipients
        self.latency_estimates = np.zeros(self.n)  # estimated latency for each recipient
        self.action_counts = np.zeros(self.n)  # how many times each recipient was chosen

    def select_recipient(self):
        # e-greedy selection
        if random.random() < self.epsilon:
            return random.randint(0, self.n - 1)  # explore: random recipient
        else:
            return np.argmin(self.latency_estimates)  # exploit: select recipient with least estimated latency

    def update_estimate(self, recipient_idx, latency):
        # update count for running avg
        self.action_counts[recipient_idx] += 1

        # inc update to latency estimate (running average)
        n = self.action_counts[recipient_idx]
        self.latency_estimates[recipient_idx] += (latency - self.latency_estimates[recipient_idx]) / n




exploration_rate = 0.1
selector = PreferPreviousRecipient(3, epsilon=exploration_rate)
latencies = [
    {"high": 100, "low": 50},   # 1
    {"high": 30, "low": 20},    # 2
    {"high": 40, "low": 10}]    # 3

for t in range(100):
    # select recipient
    recipient_idx = selector.select_recipient()

    # calc latency
    highlow = latencies[recipient_idx]
    latency = np.random.uniform(highlow["low"], highlow["high"])

    # update the selector with the observed latency
    selector.update_estimate(recipient_idx, latency)

    print(f"Selected: recipient_{recipient_idx}, Latency: {latency:.2f}, Updated estimates: {selector.latency_estimates}")
