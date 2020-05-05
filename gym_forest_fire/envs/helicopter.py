#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:57:09 2020

@author: ebecerra
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from forest_fire import ForestFire

# Implements postions and movement and fire embeding
class Helicopter(ForestFire):
    """
    Helicopter class
    simulates a Helicopter over a Firest Forest Automaton
    
    Superclass for EnvForestFire
    For more please check the documentation of EnvForestFire
    
    Examples
    --------
    >>> helicopter = Helicopter()
    >>> helicopter.render()
    >>> helicopter.movement_actions
    >>> helicopter.new_pos(7)
    >>> helicopter.render()
    """
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
        """Automaton visualization"""
        # Plot style
        sns.set_style('whitegrid')
        # Main Plot
        plt.imshow(self.grid_to_rgba(), aspect='equal')
        # Title showing Reward
        plt.title(title, **self.title_font)
        # Modify Axes
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, self.n_col, 1))
        ax.set_yticks(np.arange(0, self.n_row, 1))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, self.n_col, 1), **self.axes_font)
        ax.set_yticklabels(np.arange(0, self.n_row, 1), **self.axes_font)
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.n_col, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.n_row, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=2)
        ax.grid(which='major', color='w', linestyle='-', linewidth=0)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        # Add Helicopter Cross
        marker_style = dict(color='0.7', marker='P',
                    markersize=10, markerfacecolor='0.2')
        ax.plot(self.pos_col, self.pos_row, **marker_style)
        fig = plt.gcf()
        plt.show()
        return fig

class EnvMakerForestFire(Helicopter):
    """
    version 2.1
    Implementation of a class to generate multiple Environments
    for a Reinforcement Learning task.
    
    All the created environments follow the    
    Open AI gym API:
    
    env = EnvMakerForestFire()   
        
    env.reset()
    env.step(action)
    env.render()
    env.close()
    
    The created environment simulates a helicopter trying to extinguish a forest fire.
    The forest is simulated using a Forest Fire Automaton [Drossel and Schwabl (1992)] and
    the helicopter as a position on top of the lattice and some effect over the cells.
    At each time step the Agent has to make a decision to where in the neighborhood to move the helicopter,
    then the helicopter moves and has some influence over the destination cell,
    the effect is simply changing it to another cell type, usually from 'fire' to 'empty'
    and the reward is some function of the current state of the system,
    usually just counting cells types, multiplying for some weights (positive for trees and negative for fires) and adding up.
    
    The actions to move the helicopter are the natural numbers from 1 to 9, each representing a direction:        
    1. Left-Up
    2. Up
    3. Right-Up
    4. Right
    5. Don't move
    6. Left
    7. Left-Down
    8. Down
    9. Right-Down
    
    Forest Fire Automaton Drossel and Schwabl (1992)
    
    Three type of cells: TREE, EMPTY and FIRE
    At each time step and for each cell apply the following rules
    (order does not matter).
        * With probability f:                       Lighting Rule
            TREE turns into Fire
        * If at least one neighbor is FIRE:         Propagation Rule
            TREE turns into Fire
        * Unconditional:                            Burning Rule
            FIRE turns into EMPTY
        * With probability p:
            EMPTY turns into TREE                    Growth Rule
            
    Also two more cells were added.
    ROCK, does not interacts with anything
        Used as a true death cell
        Used on the Deterministic mode
        Used on the invariant boundary conditions
    LAKE, does not interacts with anything
        Used on other classes that inherit from ForestFire
    
    Deterministic mode: The automaton does not computes
    the Lighting and Growth rules, stops when there are
    no more FIRE cells.    
    
    Parameters
    ----------
    
    env_mode : {'stochastic', 'deterinistic'}, default='stochastic'
        Main mode of the agent.
        - 'stochastic'
        Applies all the rules of the Forest Fire Automaton and sets the optional parameters (except if manually changed):
        pos_row = ceil(n_row), pos_col = ceil(n_pos), effect = 'extinguish', freeze = ceil((n_row + n_col) / 4)                 
                 termination_type = 'continuing', ip_tree = 0.75, ip_empty = 0.25, ip_fire = 0.0, ip_rock = 0.0, ip_lake = 0.0
        - 'deterministic'
        Does not apply the stochastic rules of the Fire Forest Automaton, those are the Lighting and Growth rules.
        Also sets the following parameters:
        pos_row = ceil(n_row), pos_col = ceil(n_pos), effect = 'clearing', freeze = 0                 
        termination_type = 'no_fire', ip_tree = 0.59, ip_empty = 0.0, ip_fire = 0.01, ip_rock = 0.40, ip_lake = 0.0
    n_row : int, default=16 
        Rows of the grid.
    n_col : int, default=16
        Columns of the grid.
    p_tree : float, default=0.300
        'p' probability of a new tree.
    p_fire : float, default=0.006
        'f' probability of a tree catching fire.
    custom_grid : numpy like matrix, defult=None
        If matrix provided, it would be used as the starting lattice for the automaton
        instead of the randomly generated one. Must use the same symbols for the cells.
    pos_row : int, optional
        Row position of the helicopter.
    pos_col : int, optional
        Column position of the helicopter.
    effect : {'extinguish', 'clearing'}, optional
        Effect of the helicopter over the cells.
        - 'extinguish'
        Whenever on top: transforms fire cell to empty.
        - 'clearing'
        Whenever on top: transforms tree cell to empty.
        - 'multi'
        Combines the two previous effects.
    freeze : int, optional
        Steps the Agent can make before an Automaton actuliazation.
    termination_type : {'continuing', 'no_fire', 'steps', 'threshold'}, optional
        Termination conditions for the task.
        - 'continuing'
        A never ending task.
        - 'no_fire'
        Teminate whenever there are no more fire cells, this
        is the only value that works well with env_mode='deterministic'.
        - 'steps'
        Terminate after a fixed number of steps have been made.   
        - 'threshold'
        Terminate after a fixed number of cells have been or are fire cells.
    steps_to_termination: Steps to termination, optional, default=128
        Only valid with termination_type='steps'.
    fire_threshold : Accumulated fire cell threshold, optional, default=1024
        Only valid with termination_type='threshold'.
    reward_type : {'cells', 'hits', 'both', 'duration'}, optional, defualt='cells'
        Specifies the general behavior of the reward function.
        - 'cells'
        Reward that depends only in the lattice state of the Automaton.
        Multiplies the count of each cell type for a weight and then adds up all the results.
        - 'hits'
        Each time the Agent moves to a fire cell it gets rewarded.
        - 'both'
        Combines the two previous schemes.
        - 'duration'
        Returns the current step number as a reward, only useful for termination_type='threshold'.
    reward_tree : float, default=1.0
        How much each invidual tree cell gets rewarded.
    reward_fire : float, default=-10.0
        How much each individual fire cell gets rewarded.
    reward_empty : float, defualt=0.0
        How much each individual empty cell gets rewarded.
    reward_hit : float, defualt=10.0
        Reward when moving to a fire cell.
    tree : object, default=0.77
        Symbol to represent the tree cells.
    empty : object, default=0.66
        Symbol to represent the empty cells.
    fire : object, default=-1.0
        Symbol to represent the fire cells.
    rock : object, default=0.88
        Symbol to represent the rock cells.
    lake : object, default=0.99
        Symbol to represent the lake cells.
   ip_tree : float, optional
       Initialization probability of a tree cell.
   ip_empty : float, optional
       Initialization probability of an empty cell.
   ip_fire : float, optional
       Initialization probability of a fire cell.
       When in env_mode='deterministic', at least 1 fire cell is forced onto the grid.
   ip_rock : float, optional
       Initialization probability of a rock cell.
   ip_lake : float, optional
       Initialization probability of a lake cell.
     
    Methods
    ----------
    EnvMakerForestFire.reset()
        Initializes the environment and returns the first observation
        Input :
        Returns : 
        tuple (grid, position)
            - grid
            np array with Automaton lattice
            - position
            np array with (row, col)
            
    EnvMakerForestFire.step(action) : 
        Computes a step in the system
        Input :
            - action, int {1,2,3,4,5,6,7,8,9}
        Returns :
        {observation, reward, termination, information}
            - observation
            tuple with observations from the environment (grid, position)
            - reward
            float with the reward for the actions
            - termination
            bool True is the task has already ended, False otherwise
            - infomation
            dict with extra data about the system
    
    EnvMakerForestFire.render()
        Human friendly visualization of the system
        Input :
        Returns :
        matplotlib object
    
    EnvMakerForestFire.close()
        Closes the environment, prints a message
        Input:
        Returns:
        
    Examples
    --------
    
    Instantiation
    >>> env = EnvMakerForestFire()
    
    Starts the environment and gets first observation
    >>> grid, position = env.reset()
    
    Visualization
    >>> env.render()
    
    Performs 1 random step over the environment and assigns the results
    >>> import numpy as np
    >>> actions = list(env.movement_actions)
    >>> action = np.random.choice(actions)
    >>> obs, reward, terminated, info = env.step(action)
    >>> env.render()
    
    Closes the environment
    >>> env.close 
"""
    version = 'v2.1'
    total_hits = 0
    total_burned = 0
    total_reward = 0.0
    steps = 0
    def_steps_to_termination = 128
    def_fire_threshold = 1024
    def __init__(self, env_mode = 'stochastic',
                 n_row = 16, n_col = 16, p_tree = 0.300, p_fire = 0.006, custom_grid = None,
                 pos_row = None, pos_col = None, effect = None, freeze = None,                 
                 termination_type = None, steps_to_termination = None, fire_threshold = None,
                 reward_type = 'cells',
                 reward_tree = 1.0, reward_fire = -8.0, reward_empty = 0.0, reward_hit = 10.0,
                 tree = 0.77, empty = 0.66, fire = -1.0, rock = 0.88, lake = 0.99,
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None):
        self.env_mode = env_mode
        self.custom_grid = custom_grid
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
        self.terminated = self.is_task_terminated()
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
        init_attributes = ('env_mode','n_row','n_col','pos_row','pos_col','custom_grid',
                 'p_tree','p_fire','effect','freeze','termination_type',
                 'steps_to_termination','fire_threshold','reward_type',
                 'reward_tree','reward_fire','reward_empty','reward_hit',
                 'tree','empty','fire','rock','lake','ip_tree','ip_empty',
                 'ip_fire','ip_rock','ip_lake')       
        init_values = (self.env_mode,self.n_row,self.n_col,self.pos_row,self.pos_col, self.custom_grid,
                 self.p_tree,self.p_fire,self.effect,self.freeze,self.termination_type,
                 self.steps_to_termination,self.fire_threshold,self.reward_type,
                 self.reward_tree,self.reward_fire,self.reward_empty,self.reward_hit,
                 self.tree,self.empty,self.fire,self.rock,self.lake,self.ip_tree,self.ip_empty,
                 self.ip_fire,self.ip_rock,self.ip_lake)
        self.init_kwargs = dict(zip(init_attributes, init_values))
        self.__init__(**self.init_kwargs)
        if self.custom_grid is not None: self.grid_init_manually(self.custom_grid)
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
        self.terminated = self.is_task_terminated()
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
            return Helicopter.render(self, title=f'Mode: {self.env_mode.title()}\nReward: {self.reward}')
        except AttributeError:
            return Helicopter.render(self, title=f'Fores Fire Environment\nMode: {self.env_mode.title()}')
    def close(self):
        print('Gracefully Exiting, come back soon')
        return True
    def is_task_terminated(self):
        if self.termination_type == 'continuing':
            terminated = False
        elif self.termination_type == 'no_fire':
            self.count_cells()
            terminated = False if self.cell_counts[self.fire] != 0 else True
        elif self.termination_type == 'steps':
            if self.steps_to_termination is None: self.steps_to_termination = self.def_steps_to_termination
            terminated = False if self.steps < self.steps_to_termination else True  
        elif self.termination_type == 'threshold':
            if self.fire_threshold is None: self.fire_threshold = self.def_fire_threshold
            terminated = False if self.total_burned < self.fire_threshold else True  
        else:
            raise ValueError('Unrecognized termination parameter')
        return terminated
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
        elif self.effect == 'multi':
            if current_cell == self.fire:
                self.grid[row][col] = self.empty
            elif current_cell == self.tree:
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
        elif self.reward_type == 'duration':
            reward += self.steps
        else:
            raise ValueError('Unrecognized reward type')
        return reward
    def count_cells(self):
        cell_types, counts = np.unique(self.grid, return_counts=True)
        cell_counts = defaultdict(int, zip(cell_types, counts))
        self.cell_counts = cell_counts
        return cell_counts
