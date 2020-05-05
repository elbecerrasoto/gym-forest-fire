#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:11:20 2020

@author: ebecerra
"""

from gym_forest_fire.envs.helicopter import EnvMakerForestFire

from gym.envs.registration import register

register(
    id='ForestFire-v0',
    entry_point='gym_forest_fire.envs:ForestFireEnv0',
)
