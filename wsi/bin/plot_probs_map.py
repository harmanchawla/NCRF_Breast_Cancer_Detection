import numpy as np
from matplotlib import pyplot as plt
probs_map = np.load('probs_map/')
plt.imshow(probs_map.transpose(), vmin=0, vmax=1, cmap='jet')
plt.show()
