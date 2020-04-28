#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:57:09 2020

@author: ebecerra
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from forest_fire import ForestFire

# Implements postions and movement and fire embeding
class Helicopter(ForestFire):
    """ Helicopter class """
    movement_actions = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    def __init__(self, pos_row = None, pos_col = None,
                 n_row = 16, n_col = 16, p_tree=0.100, p_fire=0.001,
                 forest_mode = 'stochastic', force_fire = True, boundary='invariant',
                 tree = '|', empty = '.', fire = '*', rock = '#', lake = 'O',
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None):
        # Forest Fire Parameters
        ForestFire.__init__(self,n_row,n_col,p_tree,p_fire,forest_mode,
                            force_fire,boundary,tree,empty,fire,rock,lake,
                            ip_tree,ip_empty,ip_fire,ip_rock,ip_lake)
        # Helicopter attributes
        if pos_row is None:
            # Start aprox in the middle
            self.pos_row = math.ceil(self.n_row/2)
        else:
            self.pos_row = pos_row
        if pos_col is None:
            # Start aprox in the middle
            self.pos_col = math.ceil(self.n_col/2)
        else:
            self.pos_col = pos_col      
    def new_pos(self, movement):
        self.pos_row = self.pos_row if movement == 5\
            else self.pos_row if self.is_out_borders(movement, pos='row')\
            else self.pos_row - 1 if movement in [1,2,3]\
            else self.pos_row + 1 if movement in [7,8,9]\
            else self.pos_row
        self.pos_col = self.pos_col if movement == 5\
            else self.pos_col if self.is_out_borders(movement, pos='col')\
            else self.pos_col - 1 if movement in [1,4,7]\
            else self.pos_col + 1 if movement in [3,6,9]\
            else self.pos_col
        return (self.pos_row, self.pos_col)
    def is_out_borders(self, movement, pos):
        if pos == 'row':
            # Check Up movement
            if movement in [1,2,3] and self.pos_row == 0:
                out_of_border = True
            # Check Down movement
            elif movement in [7,8,9] and self.pos_row == self.n_row-1:
                out_of_border = True
            else:
                out_of_border = False
        elif pos == 'col':
            # Check Left movement
            if movement in [1,4,7] and self.pos_col == 0:
                out_of_border = True
            # Check Right movement
            elif movement in [3,6,9] and self.pos_col == self.n_col-1:
                out_of_border = True
            else:
                out_of_border = False
        else:
            raise 'Argument Error: pos = "row" | "col"'
        return out_of_border
    def render(self, title='Forest Fire Automaton'):
        def add_cross(plot_function, title):
            def wrapper():
                plt.scatter(self.pos_col, self.pos_row,
                            marker='x', color='black',
                            s=50, linewidths=50,
                            zorder=11)                
                plot_function(title=title)
            return wrapper
        new_render = add_cross(super().render, title)
        new_render()

class EnvForestFire(Helicopter):
    total_hits = 0
    total_burned = 0
    total_reward = 0.0
    steps = 0
    def_steps_to_termination = 128
    def_fire_threshold = 1024
    def __init__(self, env_mode = 'stochastic',
                 n_row = 16, n_col = 16, pos_row = None, pos_col = None,
                 p_tree=0.100, p_fire=0.001,
                 effect = None, freeze = None,                 
                 termination_type = None, steps_to_termination = None, fire_threshold = None,
                 reward_type = 'cells',
                 reward_tree = 1.0, reward_fire = -10.0, reward_empty = 0.0, reward_hit = 10.0,
                 tree = 0.3, empty = 0.1, fire = 0.7, rock = 0.02, lake = 0.9,
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None):
        self.env_mode = env_mode
        Helicopter.__init__(self,pos_row = pos_row, pos_col = pos_col,
                            n_row = n_row, n_col = n_col, p_tree = p_tree,
                            p_fire = p_fire, forest_mode = self.env_mode,
                            tree = tree, empty = empty, fire = fire, rock = rock, lake = lake,
                            ip_tree = ip_tree, ip_empty = ip_empty, ip_fire = ip_fire, ip_rock = ip_rock, ip_lake = ip_lake)
        self.init_env_mode(effect, freeze, termination_type)
        self.defrost = self.freeze
        self.steps_to_termination = steps_to_termination
        self.fire_threshold = fire_threshold
        self.reward_type = reward_type
        self.reward_tree = reward_tree
        self.reward_fire = reward_fire
        self.reward_empty = reward_empty
        self.reward_hit = reward_hit
        self.is_task_terminated()
    def init_env_mode(self, effect, freeze, termination_type):
        if self.env_mode == 'stochastic':
            self.effect = 'extinguish' if effect is None else effect
            self.freeze = math.ceil((self.n_row + self.n_col) / 4) if freeze is None else freeze
            self.termination_type = 'continuing' if termination_type is None else termination_type
        elif self.env_mode == 'deterministic':
            self.effect = 'clearing' if effect is None else effect
            self.freeze = 0 if freeze is None else freeze
            self.termination_type = 'no_fire' if termination_type is None else termination_type         
        else:
            raise ValueError('Unrecognized Environment Mode')
    def reset(self):
        init_attributes = ('env_mode','n_row','n_col','pos_row','pos_col',
                 'p_tree','p_fire','effect','freeze','termination_type',
                 'steps_to_termination','fire_threshold','reward_type',
                 'reward_tree','reward_fire','reward_empty','reward_hit',
                 'tree','empty','fire','rock','lake','ip_tree','ip_empty',
                 'ip_fire','ip_rock','ip_lake')       
        init_values = (self.env_mode,self.n_row,self.n_col,self.pos_row,self.pos_col,
                 self.p_tree,self.p_fire,self.effect,self.freeze,self.termination_type,
                 self.steps_to_termination,self.fire_threshold,self.reward_type,
                 self.reward_tree,self.reward_fire,self.reward_empty,self.reward_hit,
                 self.tree,self.empty,self.fire,self.rock,self.lake,self.ip_tree,self.ip_empty,
                 self.ip_fire,self.ip_rock,self.ip_lake)
        self.init_kwargs = dict(zip(init_attributes, init_values))
        self.__init__(**self.init_kwargs)
        self.total_hits = 0
        self.total_burned = 0
        self.total_reward = 0.0
        self.steps = 0
        self.obs = (self.grid, np.array([self.pos_row, self.pos_col]))
        return self.obs
    first_time_termination = True
    def step(self, action):
        """Must return tuple with
        numpy array, int reward, bool termination, dict info
        """
        self.steps += 1
        self.is_task_terminated()
        if not self.terminated:
            if self.defrost == 0:
                # Run fire simulation
                self.update()
                self.defrost = self.freeze
            else:
                self.defrost -= 1
            self.new_pos(action)
            self.effect_over_cells()
            self.total_hits += self.hit
            self.reward = self.calculate_reward() if self.env_mode == 'stochastic' else 0.0
        else:
            if self.env_mode == 'deterministic' and self.first_time_termination:
                self.reward = self.calculate_reward()
            else:
                self.reward = 0.0
            self.first_time_termination = False
        self.total_reward += self.reward
        self.obs = (self.grid, np.array([self.pos_row, self.pos_col]))
        self.total_burned += self.count_cells()[self.fire]
        info = {'steps': self.steps, 'total_reward': self.total_reward,
                'total_hits': self.total_hits, 'total_burned': self.total_burned}
        return (self.obs, self.reward, self.terminated, info)
    def render(self):
        try:
            Helicopter.render(self, title=f'Mode: {self.env_mode.title()}\nReward: {self.reward}')
        except AttributeError:
            Helicopter.render(self, title=f'Fores Fire Environment\nMode: {self.env_mode.title()}')
    def close(self):
        print('Gracefully Exiting, come back soon')
    def is_task_terminated(self):
        if self.termination_type == 'continuing':
            self.terminated = False
        elif self.termination_type == 'no_fire':
            self.count_cells()
            self.terminated = False if self.cell_counts[self.fire] != 0 else True
        elif self.termination_type == 'steps':
            if self.steps_to_termination is None: self.steps_to_termination = self.def_steps_to_termination
            self.terminated = False if self.steps < self.steps_to_termination else True  
        elif self.termination_type == 'threshold':
            if self.fire_threshold is None: self.fire_threshold = self.def_fire_threshold
            self.terminated = False if self.total_burned < self.fire_threshold else True  
        else:
            raise ValueError('Unrecognized termination parameter')
        return self.terminated
    def effect_over_cells(self):
        self.hit = False
        row = self.pos_row
        col = self.pos_col
        current_cell = self.grid[row][col]
        if current_cell == self.fire: self.hit = True 
        if self.effect == 'extinguish':
            if current_cell == self.fire:
                self.grid[row][col] = self.empty
        elif self.effect == 'clearing':
            if current_cell == self.tree:
                self.grid[row][col] = self.empty
        else:
            raise ValueError('Unrecognized effect over cells')
    def calculate_reward(self):          
        reward = 0.0
        self.count_cells()
        if self.reward_type == 'cells':
            reward += self.cell_counts[self.tree] * self.reward_tree
            reward += self.cell_counts[self.empty] * self.reward_empty
            reward += self.cell_counts[self.fire] * self.reward_fire
        elif self.reward_type == 'hits':
            reward += self.hit * self.reward_hit
        elif self.reward_type == 'both':
            reward += self.cell_counts[self.tree] * self.reward_tree
            reward += self.cell_counts[self.empty] * self.reward_empty
            reward += self.cell_counts[self.fire] * self.reward_fire
            reward += self.hit * self.reward_hit
        else:
            raise ValueError('Unrecognized reward type')
        return reward
    def count_cells(self):
        cell_types, counts = np.unique(self.grid, return_counts=True)
        cell_counts = defaultdict(int, zip(cell_types, counts))
        self.cell_counts = cell_counts
        return cell_counts
