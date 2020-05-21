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

# Implements positions and movement and fire embedding
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
    def __init__(self, init_pos_row = None, init_pos_col = None,
                 n_row = 16, n_col = 16, p_tree=0.100, p_fire=0.001,
                 forest_mode = 'stochastic', custom_grid = None,
                 force_fire = True, boundary='invariant',
                 tree = '|', empty = '.', fire = '*', rock = '#', lake = 'O',
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None):
        # Forest Fire Parameters
        kw_params={'n_row':n_row, 'n_col':n_col, 'p_tree':p_tree, 'p_fire':p_fire, 'forest_mode':forest_mode,
                 'custom_grid':custom_grid, 'force_fire':force_fire, 'boundary':boundary,
                 'tree':tree, 'empty':empty, 'fire':fire, 'rock':rock, 'lake':lake,
                 'ip_tree':ip_tree, 'ip_empty':ip_empty, 'ip_fire':ip_fire, 'ip_rock':ip_rock, 'ip_lake':ip_lake}
        ForestFire.__init__(self,**kw_params)
        self.init_pos_row = init_pos_row
        self.init_pos_col = init_pos_col
        # Helicopter attributes
        if init_pos_row is None:
            # Start aprox in the middle
            self.pos_row = self.n_row//2
        else:
            self.pos_row = init_pos_row
        if init_pos_col is None:
            # Start aprox in the middle
            self.pos_col = self.n_col//2
        else:
            self.pos_col = init_pos_col
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
        grid = self.grid_to_rgba()
        # Plot style
        sns.set_style('whitegrid')
        # Main Plot
        plt.imshow(grid, aspect='equal')
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
                    markersize=12, markerfacecolor='0.2')
        ax.plot(self.pos_col, self.pos_row, **marker_style)
        fig = plt.gcf()
        plt.show()
        return fig

class EnvMakerForestFire(Helicopter):
    """
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

    The observations gotten from the environment are a tuple of:

        1. 2D np-array with the current state of the cellular automaton grid.
        2. np-array with the position of the helicopter [row, col].
        3. np-array with the remaining moves of the helicopter until the next automaton update [moves].

    Parameters
    ----------

    env_mode : {'stochastic', 'deterinistic'}, default='stochastic'
        Main mode of the agent.
        - 'stochastic'
        Applies all the rules of the Forest Fire Automaton and sets the optional parameters (except if manually changed):
        pos_row = ceil(n_row), pos_col = ceil(n_pos), effect = 'extinguish', moves_before_updating = ceil((n_row + n_col) / 4)
                 termination_type = 'continuing', ip_tree = 0.75, ip_empty = 0.25, ip_fire = 0.0, ip_rock = 0.0, ip_lake = 0.0
        - 'deterministic'
        Does not apply the stochastic rules of the Fire Forest Automaton, those are the Lighting and Growth rules.
        Also sets the following parameters:
        pos_row = ceil(n_row), pos_col = ceil(n_pos), effect = 'clearing', moves_before_updating = 0
        termination_type = 'no_fire', ip_tree = 0.59, ip_empty = 0.0, ip_fire = 0.01, ip_rock = 0.40, ip_lake = 0.0
    n_row : int, default=16
        Rows of the grid.
    n_col : int, default=16
        Columns of the grid.
    p_tree : float, default=0.300
        'p' probability of a new tree.
    p_fire : float, default=0.003
        'f' probability of a tree catching fire.
    custom_grid : numpy like matrix, defult=None
        If matrix provided, it would be used as the starting lattice for the automaton
        instead of the randomly generated one. Must use the same symbols for the cells.
    pos_row : int, optional
        Row position of the helicopter.
    pos_col : int, optional
        Column position of the helicopter.
    moves_before_updating : int, optional
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
    tree : object, default=0
        Symbol to represent the tree cells.
    empty : object, default=1
        Symbol to represent the empty cells.
    fire : object, default=2
        Symbol to represent the fire cells.
    rock : object, default=3
        Symbol to represent the rock cells.
    lake : object, default=4
        Symbol to represent the lake cells.
    observation_mode : {'plain', 'one_hot', 'channels', 'channels3', 'channels4'}, default='one_hot'
        How to return the grid observation.
        - plain
        The step method returns the observation grid as a matrix of the the cells symbols.
        - one_hot
        The step method returns the observation grid as a matrix
        with entries of the cells symbols on one hot encoding. In the following way:
            tree: [1,0,0,0,0]
            empty: [0,1,0,0,0]
            fire: [0,0,1,0,0]
            rock: [0,0,0,1,0]
            lake: [0,0,0,0,1]
        - channels
        The step method returns the observation grid as a ndarray of 5 channels (5 matrices).
        A channel per cell type (5).
        On each channel, `1` marks the prescence of that cell type at that location and `0` otherwise.
        - channels3
        Same as `'channels'`, but only returns the first three channels.
        Useful when the environment will only yield tree, empty or fire cells.
        - channels4
        Same as `'channels'`, but only returns the first four channels.
        Useful when the environment will only yield tree, empty, fire or rock cells.
    sub_tree : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is 'empty'.
    sub_empty : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    sub_fire : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is 'empty'.
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    sub_rock : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is no effect (It is substituted by itself).
    sub_lake : {'tree', 'empty', 'fire', 'rock', 'lake'}, optional
        Helicopter effect over the cell. To which cell substitute current cell.
        Default in 'stochastic' mode is no effect (It is substituted by itself).
        Default in 'deterministic' mode is no effect (It is substituted by itself).
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
    # Metadata
    version = 'v2.4'
    # Global Info
    terminated = False
    first_termination = True
    total_hits = 0
    total_burned = 0
    total_reward = 0.0
    steps = 0
    # Defaults
    def_steps_to_termination = 128
    def_fire_threshold = 1024

    def __init__(self, env_mode = 'stochastic',
                 n_row = 16, n_col = 16, p_tree = 0.300, p_fire = 0.003, custom_grid = None,
                 init_pos_row = None, init_pos_col = None, moves_before_updating = None,
                 termination_type = None, steps_to_termination = None, fire_threshold = None,
                 reward_type = 'cells', reward_tree = 1.0, reward_fire = -8.0, reward_empty = 0.0, reward_hit = 10.0,
                 tree = 0, empty = 1, fire = 2, rock = 3, lake = 4, observation_mode = 'one_hot',
                 sub_tree = None, sub_empty = None, sub_fire = None, sub_rock = None, sub_lake = None,
                 ip_tree = None, ip_empty = None, ip_fire = None, ip_rock = None, ip_lake = None):

        kw_params = {
            'init_pos_row': init_pos_row, 'init_pos_col': init_pos_col,
            'n_row': n_row, 'n_col': n_col, 'p_tree': p_tree, 'p_fire': p_fire,
            'forest_mode': env_mode, 'custom_grid': custom_grid,
            'tree': tree, 'empty': empty, 'fire': fire, 'rock': rock, 'lake': lake,
            'ip_tree': ip_tree, 'ip_empty': ip_empty, 'ip_fire': ip_fire, 'ip_rock': ip_rock, 'ip_lake': ip_lake
            }

        # Helicopter Initialization
        Helicopter.__init__(self, **kw_params)

        self.env_mode = env_mode

        # Effect over cells, substitution rules
        self.sub_tree, self.sub_empty, self.sub_fire, self.sub_rock, self.sub_lake = sub_tree, sub_empty, sub_fire, sub_rock, sub_lake

        # Automatic initialization according to env_mode. It sets: moves_before_updating, termination_type and default substitutions
        self.init_env_mode(moves_before_updating, termination_type)

        # Initialization of the cell substitution rules dictionary, for the effects of the helicopter
        self.init_effects_dict()

        # Counter for updating the grid
        self.remaining_moves = self.moves_before_updating

        # Termination type specific params
        self.steps_to_termination = steps_to_termination
        self.fire_threshold = fire_threshold

        # Reward params
        self.reward_type = reward_type
        self.reward_tree = reward_tree
        self.reward_fire = reward_fire
        self.reward_empty = reward_empty
        self.reward_hit = reward_hit

        # Grid Observations and One Hot Encoding
        self.observation_mode = observation_mode
        self.onehot_translation = {self.tree: [1,0,0,0,0],
                      self.empty: [0,1,0,0,0],
                      self.fire: [0,0,1,0,0],
                      self.rock: [0,0,0,1,0],
                      self.lake: [0,0,0,0,1]}

    def init_env_mode(self, moves_before_updating, termination_type):
        if self.env_mode == 'stochastic':
            speed = math.ceil((self.n_row + self.n_col) / 4)
            self.sub_fire = 'empty' if self.sub_fire is None else self.sub_fire
            self.moves_before_updating = speed if moves_before_updating is None else moves_before_updating
            self.termination_type = 'continuing' if termination_type is None else termination_type

        elif self.env_mode == 'deterministic':
            self.sub_tree = 'empty' if self.sub_fire is None else self.sub_fire
            self.moves_before_updating = 0 if moves_before_updating is None else moves_before_updating
            self.termination_type = 'no_fire' if termination_type is None else termination_type
        else:
            raise ValueError('Unrecognized Environment Mode')

    def init_effects_dict(self):
        effect_translation={
            'tree': self.tree,
            'empty': self.empty,
            'fire': self.fire,
            'rock': self.rock,
            'lake': self.lake}
        effect_over_tree = effect_translation['tree'] if self.sub_tree is None else effect_translation[self.sub_tree]
        effect_over_empty = effect_translation['empty'] if self.sub_empty is None else effect_translation[self.sub_empty]
        effect_over_fire = effect_translation['fire'] if self.sub_fire is None else effect_translation[self.sub_fire]
        effect_over_rock = effect_translation['rock'] if self.sub_rock is None else effect_translation[self.sub_rock]
        effect_over_lake = effect_translation['lake'] if self.sub_lake is None else effect_translation[self.sub_lake]
        self.effects_dict={
            self.tree: effect_over_tree,
            self.empty: effect_over_empty,
            self.fire: effect_over_fire,
            self.rock: effect_over_rock,
            self.lake: effect_over_lake
            }

    def init_global_info(self):
        self.terminated = False
        self.total_hits = 0
        self.total_burned = 0
        self.total_reward = 0.0
        self.steps = 0
        try:
            delattr(self, 'reward')
        except AttributeError:
            pass

    def reset(self):
        self.init_kw_params = {
            'env_mode': self.env_mode,
            'n_row': self.n_row, 'n_col': self.n_col, 'p_tree': self.p_tree, 'p_fire': self.p_fire, 'custom_grid': self.custom_grid,
            'init_pos_row': self.init_pos_row, 'init_pos_col': self.init_pos_col, 'moves_before_updating': self.moves_before_updating,
            'termination_type': self.termination_type, 'steps_to_termination': self.steps_to_termination, 'fire_threshold': self.fire_threshold,
            'reward_type': self.reward_type, 'reward_tree': self.reward_tree, 'reward_fire': self.reward_fire, 'reward_empty': self.reward_empty, 'reward_hit': self.reward_hit,
            'tree': self.tree, 'empty': self.empty, 'fire': self.fire, 'rock': self.rock, 'lake': self.lake, 'observation_mode': self.observation_mode,
            'sub_tree': self.sub_tree, 'sub_empty': self.sub_empty, 'sub_fire': self.sub_fire, 'sub_rock': self.sub_rock, 'sub_lake': self.sub_lake,
            'ip_tree': self.ip_tree, 'ip_empty': self.ip_empty, 'ip_fire': self.ip_fire, 'ip_rock': self.ip_rock, 'ip_lake': self.ip_lake
            }

        # Rerun object method init
        self.__init__(**self.init_kw_params)
        # Restart global vars
        self.init_global_info()
        # Return observations, gym API
        obs_grid = self.observation_grid()
        self.obs = (obs_grid, np.array([self.pos_row, self.pos_col]), np.array([self.remaining_moves]))
        return self.obs

    def step(self, action):
        """Must return tuple with
        numpy array, int reward, bool termination, dict info
        """
        self.steps += 1

        if not self.terminated:
            # Is it time to update forest?
            if self.remaining_moves == 0:
                # Run fire simulation
                self.update()
                # Restart the counter
                self.remaining_moves = self.moves_before_updating
            else:
                self.remaining_moves -= 1

            # Move the helicopter
            self.new_pos(action)
            # Register if it has moved towards fire
            current_cell = self.grid[self.pos_row][self.pos_col]
            self.hit = True if current_cell == self.fire else False
            self.total_hits += self.hit
            # Apply the powers of the helicopter over the grid (cell substitution)
            self.effect_over_cells()
            # Calculate reward only in 'stochastic' mode
            self.reward = self.calculate_reward() if self.env_mode == 'stochastic' else 0.0

        else:
            # Calculate reward if the episode has just ended, 0.0 otherwise
            if self.first_termination:
                self.reward = self.calculate_reward()
                self.first_termination = False
            else:
                # Convert from episodic to continuing task by always returning 0.0 reward if the episode is over
                self.reward = 0.0

        # Check for stopping condition
        self.terminated = self.is_task_terminated()

        # Update some global info
        self.total_reward += self.reward
        self.total_burned += self.count_cells()[self.fire]

        # Observations for gym API
        obs_grid = self.observation_grid()
        self.obs = (obs_grid, np.array([self.pos_row, self.pos_col]), np.array([self.remaining_moves]))
        # Info for gym API
        info = {'steps': self.steps, 'total_reward': self.total_reward,
                'total_hits': self.total_hits, 'total_burned': self.total_burned}
        # Gym API
        return (self.obs, self.reward, self.terminated, info)

    def render(self):
        try:
            return Helicopter.render(self, title=f'Moves: {self.remaining_moves}\nReward: {np.round(self.reward, 4)}')
        except AttributeError:
            return Helicopter.render(self, title=f'Forest Fire Environment\nMode: {self.env_mode.title()}')

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
        row = self.pos_row
        col = self.pos_col
        current_cell = self.grid[row][col]
        # Make the substituion of current cell, following effects_dict
        for symbol in self.effects_dict:
            if symbol == current_cell:
                self.grid[row][col] = self.effects_dict[symbol]
                break

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

    def random_policy(self):
        actions = list(self.movement_actions)
        action = np.random.choice(actions)
        return action

    def observation_grid(self):
        if self.observation_mode == 'plain':
            return self.grid
        elif self.observation_mode == 'one_hot':
            return self.get_onehot_forest()
        elif self.observation_mode == 'channels':
            return self.get_channels_forest()
        elif self.observation_mode == 'channels3':
            return self.get_channels_forest()[:3]
        elif self.observation_mode == 'channels4':
            return self.get_channels_forest()[:4]
        else:
            raise ValueError("Bad Observation Mode.\nTry: {'plain', 'one_hot', 'channels'}")

    def get_onehot_forest(self):
        onehot_grid = self.grid.tolist()
        for row in range(self.n_row):
            for col in range(self.n_col):
                current_cell = onehot_grid[row][col]
                for key in self.onehot_translation:
                    if key == current_cell:
                        onehot_grid[row][col] = self.onehot_translation[key]
                        break
        return np.array(onehot_grid)

    def get_channels_forest(self):
        grid = self.get_onehot_forest()
        return np.array([grid[:,:,channel] for channel in range(np.shape(grid)[-1])])