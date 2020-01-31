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

class ForestFireEnv(helicopter.Helicopter, gym.Env):
    metadata = {'render.modes': ['human']}