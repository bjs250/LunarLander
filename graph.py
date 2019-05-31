"""
This file is used to generate plots by changing the load string
"""

import numpy as np
import matplotlib.pyplot as plt

suffix = "run - vanilla"
scores = np.load('data/test/scores - ' + suffix + '.npy')
cmr = np.load('data/test/cmr - ' + suffix + '.npy')

f1 = plt.figure(1)
line1, = plt.plot(np.arange(len(scores)), scores, ".")
line2, = plt.plot(cmr, label="cmr",  linestyle="-",color="red")

plt.legend([line1, line2], ["total reward","avg total reward"])
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Training - Vanilla')
plt.xlim([0, len(scores)])
plt.ylim([-300, 350])
plt.grid()


suffix = "run - duel"
scores = np.load('data/test/scores - ' + suffix + '.npy')
cmr = np.load('data/test/cmr - ' + suffix + '.npy')

f2 = plt.figure(2)
line3, = plt.plot(np.arange(len(scores)), scores, ".")
line4, = plt.plot(cmr, label="cmr",  linestyle="-",color="red")

plt.legend([line3, line4], ["total reward","avg total rewards"])
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Training - Duel')
plt.xlim([0, len(scores)])
plt.ylim([-300, 350])
plt.grid()
plt.show()
