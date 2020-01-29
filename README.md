# Gym Forest Fire
Forest Fire Environment for OpenAI Gym.

## Installation
1. Install [OpenAi Gym](https://github.com/openai/gym)
```bash
pip install gym
```

2. Download and install `gym-forest-fire`
```bash
git clone https://github.com/elbecerrasoto/gym-forest-fire
cd gym-forest-fire
pip install -e .
```

## Basic Info
The _ForestFire-v0_ environment implements a
[forest fire cellular automaton](https://en.wikipedia.org/wiki/Forest-fire_model)
of 16x16 cells, with parameters _f=0.001_ and _p=0.1_

The control task is to move a helicopter through the lattice,
to try to extinguish the fire.

## Running
Start by importing the package and initializing the environment
```python
import gym
import gym_forest_fire
env = gym.make('ForestFire-v0')
```

## Random Policy
Implementing the random policy
```python
import gym
import gym_forest_fire
import numpy as np

env = gym.make('ForestFire-v0')

# First observation
observation = env.reset()

# Initial lattice
env.render()

total_reward = 0
for i in range(8 * 20 + 1):
  print('.', end='')
  action = np.random.choice(np.arange(1,10))
  observation, reward, done, info = env.step(action)
  total_reward += reward
  if i%8 == 0:
      print('\n#### Iteration {} ####'.format(i))
      env.render()
env.close()

print('\nTotal Reward: {}'.format(total_reward))
```
## Images

The red cross marks the position of the helicopter.

![Seq0](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq0.svg)
**The forest at some time _t_**

![Seq1](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq1.svg)
**The forest at some time _t+1_**

![Seq2](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq2.svg)
**The forest at some time _t+2_**

![Seq3](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq3.svg)
**The forest at some time _t+3_**
