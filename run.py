""" This file is used to run models on test data"""

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from model import DQN

env = gym.make('LunarLander-v2')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

seed = 52
env.seed(seed)

total_actions = env.action_space.n
total_states = env.observation_space.shape[0]

flag = "duel"
model = DQN(total_states,total_actions, seed, flag)
model.load_state_dict(torch.load('data/test/saved_model - '+flag))

episodes = 100
steps = 1000
frames = 0

scores = list()
cmr = list()
score_bucket = deque(maxlen=100)

for episode in range(episodes):
    state = env.reset() # Restart the game
    score = 0

    for step in range(steps):
        frames += 1

        state = torch.from_numpy(state).float().unsqueeze(0)

        Qsa = model(state) # NN returns Q values for each action given a state
        Qsa = Qsa.data.numpy()
        action = np.argmax(Qsa)

        # Get the next state
        next_state, reward, done, info = env.step(action)
        score += reward
        state = next_state

        env.render()

        if done == True:
            break

    scores.append(score)
    score_bucket.append(score)
    cmr.append(np.mean(score_bucket))
    print("Episode: ", episode, "score", str(score)[0:8], "mean cumulative reward: ", str(np.mean(score_bucket))[0:8], len(score_bucket))

env.close()

suffix = "run - vanilla"
np.save('data/scores - ' + suffix + '.npy', np.asarray(scores))
np.save('data/cmr - ' + suffix + '.npy', np.asarray(cmr))
