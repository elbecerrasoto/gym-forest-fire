#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:35:08 2020

@author: ebecerra
"""

#Hi I am at:/home/ebecerra/Dissertation/gym-forest-fire

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
sys.path.append('gym_forest_fire/envs/')
import helicopter

class ForestFireEnv(helicopter.Helicopter, gym.Env):
    metadata = {'render.modes': ['human']}