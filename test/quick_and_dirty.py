# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.getcwd()

import numpy as np
import matplotlib.pyplot as plt


int(34.45)

color_tree = np.array([15, 198, 43, int(1.0*255)]) # Green RGB
color_empty = np.array([255, 245, 166, int(0.4*255)]) # Brown RGB
color_fire = np.array([255, 106, 58, int(1.0*255)]) # Red RGB

cells_to_colors = {'tree':color_tree,
                  'empty':color_empty,
                  'fire':color_fire}

cells = np.array(['tree','empty','fire'])

n_row = 10
n_col = 10
grid = np.random.choice(cells,
                 n_row*n_col,
                 p=[0.75, 0.15, 0.10]).\
                 reshape(n_row,n_col)

def grid_to_rgb(grid, translation_dictionary):
    n_row = grid.shape[0]
    n_col = grid.shape[1]
    grid = grid.tolist()
    for row in range(n_row):
        for col in range(n_col):
            for key in translation_dictionary:
                if str(grid[row][col]) == key:
                    grid[row][col] = translation_dictionary[key]
    return np.array(grid)

rgb_array = grid_to_rgb(grid, cells_to_colors)
reward = -20


import seaborn as sns
#plt.style.use('gadfly')
#plt.style.use('ggplot')
#plt.style.use('classic')
sns.set_style("whitegrid")
plt.imshow(rgb_array, aspect='equal', interpolation="nearest")
#plt.axis('off')
plt.title('Reward {}'.format(reward))
ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(0, 10, 1));
ax.set_yticks(np.arange(0, 10, 1));

# Labels for major ticks
ax.set_xticklabels(np.arange(0, 10, 1));
ax.set_yticklabels(np.arange(0, 10, 1));

# Minor ticks
ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
ax.set_yticks(np.arange(-.5, 10, 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=2)
ax.grid(which='major', color='w', linestyle='-', linewidth=0)
ax.tick_params(axis=u'both', which=u'both',length=0)


#### Internet Example ####
# https://stackoverflow.com/questions/38973868/adjusting-gridlines-and-ticks-in-matplotlib-imshow#38994970




