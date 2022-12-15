import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())
run_path = os.path.join("procgen_experiments", "procgen_fork", "scripts_workstation", "utils", "results", "test", "202212-0619-2931", "2-1-1", "dic.npy")

data = np.load(run_path, allow_pickle = True)

data = data.item()

window = 200
average_data = np.zeros(data['rew'].shape[0]-window)
for ind in range((average_data.shape[0]-window + 1)):
    average_data[ind] = np.mean(data['rew'][ind:ind+window])

plt.plot(average_data)
plt.show()

