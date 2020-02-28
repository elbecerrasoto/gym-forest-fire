#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:16:03 2020

@author: ebecerra
"""

# Math
import numpy as np
import math

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Dependencies
import forest_fire

# Features to add

# Action info attribute, DONE
# Rocks, dead cells
# Rock boundary
# Custom starting positions
# Water feature

class Helicopter(forest_fire.ForestFire):
    """ Helicopter class """
    def __init__(self, pos_row = None, pos_col = None, freeze = None, water = 100,
                 n_row = 16, n_col = 16,
                 p_tree=0.100, p_fire=0.001, p_init_tree=0.75,
                 boundary='reflective', tree = 3, empty = 1, fire = 7):
        super().__init__(n_row,n_col,
             p_tree,p_fire,p_init_tree,
             boundary,tree,empty,fire)
        # Helicopter attributes
        self.actions_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.actions_cardinality = len(self.actions_set)
        if pos_row is None:
            # Start aprox in the middle
            self.pos_row = math.ceil(self.n_row/2) - 1
        else:
            self.pos_row = pos_row
        if pos_col is None:
            # Start aprox in the middle
            self.pos_col = math.ceil(self.n_col/2) - 1
        else:
            self.pos_col = pos_col      
        if freeze is None:
            # Move aprox a 1/4 of the grid
            self.freeze = math.ceil((self.n_row + self.n_col) / 4)
        else:
            self.freeze = freeze
        self.defrost = self.freeze
        # Water not yet implemented
        self.water = water
        # Number of Times where a fire was extinguished
        self.hits = 0
        # Render Info
        self.current_reward = 0
        self.color_tree = np.array([15, 198, 43, int(1.0*255)]) # Green RGBA
        self.color_empty = np.array([255, 245, 166, int(1.0*255)]) # Beige RGBA
        self.color_fire = np.array([255, 106, 58, int(1.0*255)]) # Red RGBA
        self.grid_to_rgba()
    def step(self, action):
        """Must return tuple with
        numpy array, int reward, bool termination, dict info
        """
        termination = False
        if self.defrost != 0:
            self.new_pos(action)
            self.hits += self.extinguish_fire()
            self.defrost -= 1
            obs = (self.grid, np.array([self.pos_row, self.pos_col]))
            # Don't delay the reward
            # Reward it often
            reward = self.calculate_reward()
            self.current_reward = reward
            return (obs, reward, termination, {'hits': self.hits})
        if self.defrost == 0:
            # Run fire simulation
            self.update()
            self.new_pos(action)
            self.hits += self.extinguish_fire()
            self.defrost = self.freeze
            obs = (self.grid, np.array([self.pos_row, self.pos_col]))
            reward = self.calculate_reward()
            self.current_reward = reward
            return ((obs, reward, termination, {'hits': self.hits}))
    def calculate_reward(self):
        reward = 0
        for row in range(self.n_row):
            for col in range(self.n_col):
                if self.grid[row][col] == self.fire:
                    reward -= 1
        return reward
    def new_pos(self, action):
        self.pos_row = self.pos_row if action == 5\
            else self.pos_row if self.is_out_borders(action, pos='row')\
            else self.pos_row - 1 if action in [1,2,3]\
            else self.pos_row + 1 if action in [7,8,9]\
            else self.pos_row
        self.pos_col = self.pos_col if action == 5\
            else self.pos_col if self.is_out_borders(action, pos='col')\
            else self.pos_col - 1 if action in [1,4,7]\
            else self.pos_col + 1 if action in [3,6,9]\
            else self.pos_col
        return (self.pos_row, self.pos_col)
    def is_out_borders(self, action, pos):
        if pos == 'row':
            # Check Up movement
            if action in [1,2,3] and self.pos_row == 0:
                out_of_border = True
            # Check Down movement
            elif action in [7,8,9] and self.pos_row == self.n_row-1:
                out_of_border = True
            else:
                out_of_border = False
        elif pos == 'col':
            # Check Left movement
            if action in [1,4,7] and self.pos_col == 0:
                out_of_border = True
            # Check Right movement
            elif action in [3,6,9] and self.pos_col == self.n_col-1:
                out_of_border = True
            else:
                out_of_border = False
        else:
            raise "Argument Error: pos = str 'row'|'col'"
        return out_of_border
    def extinguish_fire(self):
        """Check where the helicopter is
        then extinguish at that place"""
        hit = 0
        row = self.pos_row
        col = self.pos_col
        current_cell = self.grid[row][col]
        if current_cell == self.fire:
            self.grid[row][col] = self.empty
            hit = 1
        return hit
    def reset(self):
        # Another random grid
        self.__init__(None,None,
                      self.freeze,self.water,
                      self.n_row,self.n_col,
                      self.p_tree,self.p_fire,self.p_init_tree,
                      self.boundary,self.tree,self.empty,self.fire)
        # Return first observation
        return (self.grid, np.array([self.pos_row,self.pos_col]))
    def grid_to_rgba(self):
        rgba_mat = self.grid.tolist()
        for row in range(self.n_row):
            for col in range(self.n_col):
                if rgba_mat[row][col] == self.tree:
                    rgba_mat[row][col] = self.color_tree
                elif rgba_mat[row][col] == self.empty:
                    rgba_mat[row][col] = self.color_empty
                elif rgba_mat[row][col] == self.fire:
                    rgba_mat[row][col] = self.color_fire
                else:
                    raise 'Error: unidentified cell'
        rgba_mat = np.array(rgba_mat)
        self.rgba_mat = rgba_mat
        return rgba_mat
    def render(self):
        # Plot style
        sns.set_style('whitegrid')
        # Main Plot
        plt.imshow(self.grid_to_rgba(), aspect='equal')
        # Title showing Reward
        plt.title('Reward {}'.format(self.current_reward))
        # Modify Axes
        ax = plt.gca();
        # Major ticks
        ax.set_xticks(np.arange(0, self.n_col, 1));
        ax.set_yticks(np.arange(0, self.n_row, 1));
        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, self.n_col, 1));
        ax.set_yticklabels(np.arange(0, self.n_row, 1)); 
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.n_col, 1), minor=True);
        ax.set_yticks(np.arange(-.5, self.n_row, 1), minor=True);
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=2)
        ax.grid(which='major', color='w', linestyle='-', linewidth=0)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        # Mark the helicopter position
        # Put a red X
        plt.scatter(self.pos_col, self.pos_row,
                    marker='x', color='red',
                    s=50, linewidths=50,
                    zorder=11)
        fig = plt.gcf()
        plt.show()
        return fig
    def close():
        print('Gracefully Exiting, come back soon')
