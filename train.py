"""
Use this file to train a model using the baseline parameters
"""

import gym
import random
import time
import torch
import torch.nn.functional
import torch.optim as optim
import numpy as np
from collections import deque
from model import DQN

# start the clock
start_time = time.time()

env = gym.make('LunarLander-v2')

seed = 124
env.seed(seed)

total_actions = env.action_space.n
total_states = env.observation_space.shape[0]

gamma = 0.99

# Rainbow parameters
adam_learning_rate = 0.000625
epsilon = 1
epsilon_max = 1
epsilon_min = .01

decay_rate = 0.005 # choosen to reach min in around 400-500 episodes

tau = .001 # for soft update, taken from (Continuous control with deep reinforcement learning)

buffer_size = 1000  # replay buffer size
memory_size = 72        # minimum memories needed for training
replaybuffer = deque(maxlen=buffer_size)

flag = "duel"
suffix = "duel"
model = DQN(total_states,total_actions, seed, flag)
target = DQN(total_states,total_actions, seed, flag)
optimizer = optim.Adam(model.parameters(), lr=adam_learning_rate)

max_steps = 1000
max_episodes = 2000
episode = 0
frames = 0
target_update_rate = 4

scores = list()
cmr = list()
score_bucket = deque(maxlen=100)
score_bucket.append(0)
runtimes = list()

while episode < max_episodes and np.mean(score_bucket) < 200:
    state = env.reset() # Restart the game
    score = 0

    for step in range(max_steps):
        frames += 1

        # look at memories and train every few steps
        if len(replaybuffer) > memory_size and step % target_update_rate == 0:

            # Really bad spaghetti code to get random memories (without replacement) from the replay buffer
            memories = random.sample(replaybuffer, k = memory_size)
            states = torch.from_numpy(np.vstack([mem[0] for mem in memories])).float()
            actions = torch.from_numpy(np.vstack([mem[1] for mem in memories])).long()
            rewards = torch.from_numpy(np.vstack([mem[2] for mem in memories])).float()
            next_states = torch.from_numpy(np.vstack([mem[3] for mem in memories])).float()
            dones = torch.from_numpy(np.vstack([mem[4] for mem in memories]).astype(np.uint8)).float()

            # Double DQN update rule and strategy
            Q_model = model(states).gather(1, actions)
            Q_model_next_actions = np.argmax(model(next_states).detach(), 1).unsqueeze(1)
            Q_target = rewards + (gamma * target(next_states).gather(1, Q_model_next_actions) * (1 - dones))

            # Compute the TD error
            loss = torch.nn.functional.mse_loss(Q_model, Q_target)

            # Boilerplate torch code
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Do the soft target update
            paramlist = list()
            for i,param in enumerate(model.parameters()):
                paramlist.append(param)

            for i,tparam in enumerate(target.parameters()):
                tparam.data.copy_(tau * paramlist[i].data + (1-tau) * tparam.data)

        # Handle epsilon-greedy exploration
        state = torch.from_numpy(state).float().unsqueeze(0)
        model.eval()
        with torch.no_grad():
            Qsa = model(state)

        model.train()

        # Handle exploration/exploitation
        rand = random.uniform(0,1)
        if rand < epsilon: # Explore
            action =  random.choice(np.arange(total_actions))
        else: # Exploit
            action = np.argmax(Qsa.data.numpy())

        # Get the next state
        next_state, reward, done, info = env.step(action)
        score += reward
        mem = (state, action, reward, next_state, done)
        replaybuffer.append(mem)
        state = next_state

        if done == True:
            break

    # get mean score and display metrics to console
    scores.append(score)
    score_bucket.append(score)
    elapsed_time = time.time() - start_time
    runtimes.append(elapsed_time)
    cmr.append(np.mean(score_bucket))
    print("Episode: ", episode, "mean cumulative reward: ", str(np.mean(score_bucket))[0:8], "Epsilon: ", str(epsilon)[0:5], "Runtime (min): ", str(elapsed_time/60)[0:4])

    # epsilon decay
    epsilon = epsilon_min + (epsilon_max - epsilon_min)*np.exp(-decay_rate*episode)
    epsilon = max(epsilon_min, epsilon)
    episode += 1

    # save a backup
    if episode % 100 == 0:
        print("Saving backup")
        torch.save(model.state_dict(), 'backup.pth' + suffix)

env.close()

# Save the model and the cmr for later use
torch.save(model.state_dict(), 'saved_model - ' + suffix)
np.save('data/scores - ' + suffix + '.npy', np.asarray(scores))
np.save('data/cmr - ' + suffix + '.npy', np.asarray(cmr))
np.save('data/times - ' + suffix + '.npy', np.asarray(runtimes))
