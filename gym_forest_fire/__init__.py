#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:11:20 2020

@author: ebecerra
"""

from gym.envs.registration import register

register(
    id='forest-fire',
    entry_point='gym_forest_fire.envs:ForestFireEnv',
)
