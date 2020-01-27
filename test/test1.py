#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:37:09 2019

@author: ebecerra
"""

# Add previous directory to PATH
# to import helicopter
import sys
sys.path.insert(1, '../module/')

import time
import numpy as np
import helicopter

env = helicopter.Helicopter()
obs1 = env.reset()
total_reward = 0

# Initial testing
# Basic movement, rewards and out of borders
for i in range(10):
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
    time.sleep(0.5)




