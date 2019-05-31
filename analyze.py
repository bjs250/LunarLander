""" File is used to extract info from hyperparameter runs"""

import numpy as np
import matplotlib.pyplot as plt
prefix = "cmr - "
seeds = ["121","122","123","125","126","127"]
total_seeds = len(seeds)
tunes = ["0.85","0.9","0.95","1.05","1.1","1.15"]
total_tunes = len(tunes)
suffix = ".npy"

colors = ["b","g","r","c","m","k"]
markers = [".",",","o","v","^","<"]
lines = ["-","-","--","-","-","-"]

params = ["tau", "ms", "dr"]
total_params = len(params)

stab_score = np.zeros((total_params,total_seeds, total_tunes))

for pi,param in enumerate(params):
    print("\n" + param, pi)
    f = plt.figure(pi)
    graphlines = list()
    legendlabel = list()
    i = 0
    wins = 0
    R1exit = 0
    R2exit = 0
    for si,seed in enumerate(seeds):
        for ti,tune in enumerate(tunes):
            filename = prefix + seed + param + tune + suffix
            i = i + 1
            try:
                cmr = np.load('data/'+param+'/' + filename)
                print(i, cmr[-1], len(cmr))
                if (seed == "125" or seed == "126") and (tune == "0.95" or tune == "1.1"): #or seed == "127":
                    #marker = markers[ti], markersize = 5,
                    line, = plt.plot(cmr, label="cmr",  linestyle=lines[ti], color=colors[si])
                    graphlines.append(line)
                    legendlabel.append("seed: " + seed + "," + tune)
                if cmr[-1] > 200:
                    wins += 1
                    stab_score[pi,si,ti] = 1
                elif len(cmr) > 350 and cmr[-1] < 0:
                    R1exit += 1
                else:
                    R2exit += 1
            except:
                print("no file")
    print("wins:", wins, "Rule 1 exits:", R1exit, "Rule 2 exits:", R2exit)
    plt.legend(graphlines, legendlabel)
    plt.xlabel('episode')
    plt.ylabel('reward')
    if param == "tau":
        plt.title("Tau")
        plt.xlim([0, 350])
        plt.ylim([-250, 000])
    if param == "ms":
        plt.title("Replay Buffer Memory Size")
        plt.xlim([0, 600])
        plt.ylim([-250, 200])
    if param == "dr":
        plt.title("Epsilon Decay Rate")
        plt.xlim([0, 600])
        plt.ylim([-250, 200])
    plt.grid()

param = "self"
print("\n" + param)
f = plt.figure(4)
i = 0
wins = 0
R1exit = 0
R2exit = 0
for si,seed in enumerate(seeds):
    filename = prefix + seed + param + suffix
    i = i + 1
    try:
        cmr = np.load('data/'+param+'/' + filename)
        print(i, cmr[-1], len(cmr))
        if seed == "125" or seed == "126" or seed == "127":
            #marker = markers[ti], markersize = 5,
            line, = plt.plot(cmr, label="cmr",  linestyle=lines[si], color=colors[si])
        if cmr[-1] > 200:
            wins += 1
            stab_score[pi,si,ti] = 1
        elif len(cmr) > 350 and cmr[-1] < 0:
            R1exit += 1
        else:
            R2exit += 1
    except:
        print("no file")
print("wins:", wins+1, "Rule 1 exits:", R1exit, "Rule 2 exits:", R2exit)
plt.xlabel('episode')
plt.ylabel('reward')
plt.title(param)
plt.xlim([0, 600])
plt.ylim([-250, 200])
plt.grid()

plt.show()

stab_score_bv = 1.0/total_seeds
stab_score_b = np.ones((1,total_seeds)) * stab_score_bv
print(stab_score[0].sum(axis=0)/total_seeds)
print(stab_score[1].sum(axis=0)/total_seeds)
print(stab_score[2].sum(axis=0)/total_seeds)

e = np.zeros((total_params,1, total_seeds))
e[0] = ((stab_score[0].sum(axis=0)/total_seeds)-stab_score_b)/np.array([.15,.10,.05,.05,.10,.15])
e[1] = ((stab_score[1].sum(axis=0)/total_seeds)-stab_score_b)/np.array([.15,.10,.05,.05,.10,.15])
e[2] = ((stab_score[2].sum(axis=0)/total_seeds)-stab_score_b)/np.array([.15,.10,.05,.05,.10,.15])

o = np.zeros((3,6))
o[0] = e[0][0]
o[1] = e[1][0]
o[2] = e[2][0]
np.savetxt("elas.csv", o, delimiter=",")
