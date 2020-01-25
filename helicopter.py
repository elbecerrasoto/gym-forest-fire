#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class Helicopter

pos_row = None
pos_col = None
freeze = 4
water = 100
n_row = 10
n_col = 20
p_tree=0.1
p_fire=0.01
p_init_tree=0.85
boundary='reflective'
tree = '|'
empty = '.'
fire = '*'

############
Methods    #  
############
step(action)


Created on Sun Dec 29 11:36:25 2019
@author: ebecerra
"""

import numpy as np
import math
import time

class ForestFire():
    """ Forest class """
    def __init__(self, n_row = 10, n_col = 20,
                 p_tree=0.1, p_fire=0.01, p_init_tree=0.85,
                 boundary='reflective', tree = '|', empty = '.', fire = '*'):      
        self.n_row = n_row
        self.n_col = n_col
        self.p_tree = p_tree
        self.p_fire = p_fire
        self.p_init_tree = p_init_tree
        self.boundary = boundary
        self.tree = tree
        self.empty = empty
        self.fire =  fire
        # Declare the grid, random initialization
        self.grid = self.random_grid(self.n_row, self.n_col, self.p_init_tree)
    def __repr__(self):
        print('Tree represented as: {}'.format(self.tree))
        print('Empty represented as: {}'.format(self.empty))
        print('Fire Object represented as: {}'.format(self.fire))
        return str(self.grid)
    def reset(self):
        self.grid = self.random_grid(self.n_row, self.n_col, self.p_init_tree)
    def random_grid(self, n_row, n_col, p_init_tree = 0.85):
        prob_empty = 1 - p_init_tree
        grid = np.random.choice([self.tree, self.empty],
                            n_row*n_col, p=[p_init_tree, prob_empty]).\
                            reshape(n_row,n_col)
        return grid
    def fire_around(self, grid, row, col, boundary = 'reflective'):
        """ Checks for fire in the Neighborhood of grid[row][col]"""
        if boundary == 'reflective':
            neighborhood = self.neighborhood_reflective(grid, row, col)
        elif boundary == 'toroidal':
            neighborhood = self.neighborhood_toroidal(grid, row, col)
        else:
            raise Exception('Bad Boundary Name')
        burning_near = False
        for neighbor_state in neighborhood: 
            if neighbor_state == self.fire:
                burning_near = True
                break
        return burning_near
    def neighborhood_reflective(self, grid, row, col):
        """Reflective Boundary Conditions"""
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        def is_bound_legal(grid, row, col):
            n_row = grid.shape[0]
            n_col = grid.shape[1]
            r_offset = row + np.array([-1, 1])
            c_offset = col + np.array([-1, 1])
            up = r_offset[0] >= 0
            down = r_offset[1] <= n_row-1
            left = c_offset[0] >= 0
            right = c_offset[1] <= n_col-1
            return {'up': up, 'down': down, 'left': left, 'right': right}
        legal_bounds = is_bound_legal(grid, row, col)
        up_left = [grid[(row-1),(col-1)]]\
            if legal_bounds['up'] and legal_bounds['left']\
            else [grid[(row-1),(col)]] if legal_bounds['up']\
            else [grid[(row),(col-1)]] if legal_bounds['left']\
            else [grid[(row),(col)]] 
        up_center = [grid[(row-1),col]]\
            if legal_bounds['up']\
            else [grid[(row),col]]
        up_right = [grid[(row-1)%n_row,(col+1)%n_col]]\
            if legal_bounds['up'] and legal_bounds['right']\
            else [grid[(row-1),(col)]] if legal_bounds['up']\
            else [grid[(row),(col+1)]] if legal_bounds['right']\
            else [grid[(row),(col)]]
        middle_left = [grid[row,(col-1)%n_col]]\
            if legal_bounds['left']\
            else [grid[(row),col]]
        middle_right = [grid[row,(col+1)%n_col]]\
            if legal_bounds['right']\
            else [grid[(row),col]]
        down_left = [grid[(row+1)%n_row,(col-1)%n_col]]\
            if legal_bounds['down'] and legal_bounds['left']\
            else [grid[(row+1),(col)]] if legal_bounds['down']\
            else [grid[(row),(col-1)]] if legal_bounds['left']\
            else [grid[(row),(col)]]
        down_center = [grid[(row+1)%n_row,col]]\
            if legal_bounds['down']\
            else [grid[(row),col]]
        down_right = [grid[(row+1)%n_row,(col+1)%n_col]]\
            if legal_bounds['down'] and legal_bounds['right']\
            else [grid[(row+1),(col)]] if legal_bounds['down']\
            else [grid[(row),(col+1)]] if legal_bounds['right']\
            else [grid[(row),(col)]]
        neighborhood = up_left + up_center + up_right +\
            middle_left + middle_right +\
            down_left + down_center + down_right 
        return neighborhood       
    def neighborhood_toroidal(self, grid, row, col):
        """Periodic Boundary Conditions (toroidal)"""
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        up_left = [grid[(row-1)%n_row, (col-1)%n_col]]
        up_center = [grid[(row-1)%n_row, col]]
        up_right = [grid[(row-1)%n_row, (col+1)%n_col]]
        middle_left = [grid[row, (col-1)%n_col]]
        middle_right = [grid[row, (col+1)%n_col]]
        down_left = [grid[(row+1)%n_row, (col-1)%n_col]]
        down_center = [grid[(row+1)%n_row, col]]
        down_right = [grid[(row+1)%n_row, (col+1)%n_col]]
        neighborhood = up_left + up_center + up_right +\
            middle_left + middle_right +\
            down_left + down_center + down_right
        return neighborhood
    def update(self, grid=None, p_tree=None, p_fire=None):
        """ p is probability of a new tree
        f is probability of a tree catching fire """
        if grid is None:
            grid = self.grid
        if p_tree is None:
            p_tree = self.p_tree
        if p_fire is None:
            p_fire = self.p_fire
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        new_grid = grid.copy()
        for row in range(n_row):
            for col in range(n_col):
                if grid[row][col] == self.tree and self.fire_around(grid, row, col, self.boundary):
                    # Burn tree to the ground
                    new_grid[row][col] = self.fire
                elif grid[row][col] == self.tree:
                    # Roll a dice for a lightning strike
                    strike = np.random.choice([True, False], 1, p=[p_fire, 1-p_fire])[0]
                    if strike:
                       new_grid[row][col] = self.fire 
                elif grid[row][col] == self.empty:
                    # Roll a dice for a growing bush
                    growth = np.random.choice([True, False], 1, p=[p_tree, 1-p_tree])[0]
                    if growth:
                       new_grid[row][col] = self.tree
                elif grid[row][col] == self.fire:
                    # Consume fire
                    new_grid[row][col] = self.empty
                else:
                    pass
        self.grid = new_grid
        return new_grid
    def simulate(self):
        i = 1
        print('\n#### Iteration {} ####'.format(i), end='\n\n')
        print(self.grid)
        while True:
            i += 1
            print('\n#### Iteration {} ####'.format(i), end='\n\n')
            self.update()
            print(self.grid)
            time.sleep(0.7)

class Helicopter(ForestFire):
    """ Helicopter class """
    def __init__(self, pos_row = None, pos_col = None, freeze = 4, water = 100,
                 n_row = 10, n_col = 20,
                 p_tree=0.1, p_fire=0.01, p_init_tree=0.85,
                 boundary='reflective', tree = '|', empty = '.', fire = '*'):
        super().__init__(n_row,n_col,
             p_tree,p_fire,p_init_tree,
             boundary,tree,empty,fire)
        # Helicopter attributes
        if pos_row is None:
            self.pos_row = math.ceil(self.n_row/2)
        if pos_col is None:
            self.pos_col = math.ceil(self.n_col/2)
        self.water = water
        self.freeze = freeze
        self.defrost = freeze
        self.hits = 0
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
            reward = self.calculate_reward()
            return (obs, reward, termination, {'hits': self.hits})
        if self.defrost == 0:
            self.new_pos(action)
            self.hits += self.extinguish_fire()
            # Run fire simulation
            self.update()
            self.defrost = self.freeze
            obs = (self.grid, np.array([self.pos_row, self.pos_col]))
            reward = self.calculate_reward()
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
        self.__init__(self.pos_row,self.pos_col,
                      self.freeze,self.water,
                      self.n_row,self.n_col,
                      self.p_tree,self.p_fire,self.p_init_tree,
                      self.boundary,self.tree,self.empty,self.fire)
        # Return first observation
        return (self.grid, np.array([self.pos_row,self.pos_col]))
    def render(self):
        pass
    def close(self):
        pass

