import numpy as np
from matplotlib import pyplot as plt

toy_sample = ['patient_000_node_3.npy', #0, negative
              'patient_005_node_3.npy', #1, itc
              'patient_007_node_4.npy', #2, micro
              'patient_013_node_3.npy', #3, macro
              'patient_020_node_1.npy', #4, micro
              'patient_020_node_3.npy', #5, negative
              'patient_020_node_4.npy', #6, macro
              'patient_021_node_2.npy', #7, negative
              'patient_100_node_0.npy', #8, negative, test
              'patient_105_node_3.npy', #9, itc, test
              'patient_107_node_4.npy', #10, micro, test
              'patient_112_node_0.npy' #11, macro, test
              ]

filename = toy_sample[6]
tissue_mask = np.load('tissue_mask/' + filename)
probs_map = np.load('probs_map/' + filename)

fig = plt.figure()
fig1 = fig.add_subplot(1, 2, 1)
fig1.imshow(tissue_mask.transpose(), vmin=0, vmax=1, cmap='jet')

fig2 = fig.add_subplot(1, 2, 2)
fig2.imshow(probs_map.transpose(), vmin=0, vmax=1, cmap='jet')

plt.show()
