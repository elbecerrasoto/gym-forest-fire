#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:35:08 2020

@author: ebecerra
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import helicopter

class ForestFireEnv(helicopter.Helicopter, gym.Env):
    pass
