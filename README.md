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
to try to extinguish the fire. The helicopter
has the effect of turning fire cells to empty cells whenever on top of them.

The possible actions to take are 9, either moving one-step into 8 directions,
or staying at the same place.

Each number from 1 to 9 represents one direction.

1. Left-Up
2. Up
3. Right-Up
4. Right
5. Don't move
6. Left
7. Left-Down
8. Down
9. Right-Down

The helicopter can move 8 times before the next computation
of the forest fire automaton. Basically, the helicopter can
travel half the distance of the forest before the next actualization.
This roughly represents the helicopter's speed.

The reward scheme is -1 per burning tree at each time.

The task is continuing.

The representation of the lattice is
a numpy character matrix.

* '|' represents a tree
* '.' represents empty
* '*' represents fire

The starting position of the helicopter is 7,7. Just in the middle.
The starting forest configuration is random,
with 0.75 chance of a tree and 0.15 of an empty space.

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
Some camptures of env.render()
The red cross marks the position of the helicopter.

![A](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq0.svg)
**The forest at some time _t_**

![B](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq1.svg)
**The forest at some time _t+1_**

![C](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq2.svg)
**The forest at some time _t+2_**

![D](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq3.svg)
**The forest at some time _t+3_**
