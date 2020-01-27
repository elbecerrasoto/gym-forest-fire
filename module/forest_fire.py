#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:46:22 2020

@author: ebecerra
"""

import numpy as np
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
    def simulate(self, times=10, delay=0.7):
        i = 1
        print('\n#### Iteration {} ####'.format(i), end='\n\n')
        print(self.grid)
        while i <= times:
            i += 1
            print('\n#### Iteration {} ####'.format(i), end='\n\n')
            self.update()
            print(self.grid)
            time.sleep(0.7)