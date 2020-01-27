#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:37:09 2019

@author: ebecerra
"""

# Main working directory
# '/home/ebecerra/Dissertation/gym-forest-fire'

# Add previous directory to PATH
# to import helicopter
import sys
sys.path.insert(1, 'module/')

import time
import numpy as np
import helicopter

env = helicopter.Helicopter()
obs1 = env.reset()
total_reward = 0

env.render()

# Initial testing
# Basic movement, rewards and out of borders
for i in range(50):
    print('\n#### Iteration {} ####'.format(i))
    action = np.random.choice(np.arange(1,10))
    print('OLD POSITION: pos row [{}], pos col[{}]'.format(env.pos_row, env.pos_col))
    print('Action Selected: {}'.format(action))
    print('Is out ROW {}'.format(env.is_out_borders(action, pos='row')))
    print('Is out COL {}'.format(env.is_out_borders(action, pos='col')))
    res = env.step(action)
    reward = res[1]
    total_hits = res[3]['hits']
    total_reward += reward
    print('NEW POSITION: pos row [{}], pos col[{}]'.format(env.pos_row, env.pos_col))
    print('Total Reward: {}'.format(total_reward))
    print('Total Hits: {}'.format(total_hits))
    if i % env.freeze == 0:
        env.render()
    time.sleep(0.4)
    

# Design choices:
    # If runned interactively with render use Ipython
    # If model, do nothing

#import os
#os.getcwd()
#    def random_policy(self, iterations=100, delay=0.5):
#        total_reward = 0
#        self.render()
#        for i in range(iterations):
#            print('\n#### Iteration {} ####'.format(i))
#            action = np.random.choice(np.arange(1,10))
#            print('OLD POSITION: pos row [{}], pos col[{}]'.format(self.pos_row, self.pos_col))
#            print('Action Selected: {}'.format(action))
#            out = self.step(action)
#            reward = out[1]
#            total_hits = out[3]['hits']
#            total_reward += reward
#            print('NEW POSITION: pos row [{}], pos col[{}]'.format(self.pos_row, self.pos_col))
#            print('Reward: {}'.format(reward))
#            print('Total Reward: {}'.format(total_reward))
#            print('Total Hits: {}'.format(total_hits))
#            self.render()
#            time.sleep(delay)

