#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:35:08 2020

@author: ebecerra
"""

import os
directory = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.insert(1, directory)

import helicopter

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class ForestFireEnv0(gym.Env):
    metadata = {'render.modes': ['human']}
    # Parameters
    maker_params = {'env_mode': 'stochastic',
                 'n_row': 16, 'n_col': 16, 'p_tree': 0.100, 'p_fire': 0.005,
                 'init_pos_row': 8, 'init_pos_col': 8,
                 'termination_type': 'continuing', 'reward_type': 'cells', 'observation_mode': 'plain'}
    cell_symbols = {'tree': 0.77, 'empty': 0.66, 'fire': -1.0, 'rock': 0.88, 'lake': 0.99}
    substitution_effects = {'sub_tree': None, 'sub_empty': None, 'sub_fire': 'empty', 'sub_rock': None, 'sub_lake': None}
    reward_values = {'reward_tree': 0.0, 'reward_fire': -1.0, 'reward_empty': 0.0, 'reward_hit': None,}
    init_cell_probs = {'ip_tree': 0.75, 'ip_empty': 0.25, 'ip_fire': None, 'ip_rock': 0.00, 'ip_lake': None}
    # Instantiated Environment
    env = helicopter.EnvMakerForestFire(**maker_params,
                                        **cell_symbols,
                                        **substitution_effects,
                                        **reward_values,
                                        **init_cell_probs)
    def reset(self):
    	return self.env.reset()
    def step(self, action):
    	return self.env.step(action)
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()
