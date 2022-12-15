import numpy as np
import os
print(os.getcwd())
data = np.load('utils/results/test/202212-0317-1859/2-1-0.99/dic.npy', allow_pickle = True).item()

print(data['rew'][:100])
print(data['rew'][19900:])