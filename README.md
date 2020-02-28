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
a numpy matrix.

* 3 represents a tree
* 1 represents empty
* 7 represents fire

The observation returned by the the step method is a tuple of two elements,
the first is the lattice and the second element is the postion of the helicopter in a [row, col] format.

The starting position of the helicopter is 7,7, just in the middle.
The starting forest configuration is random,
with 0.75 chance of a tree and 0.15 of an empty space.

The cell numeration starts from the left and upper corner. So the cell at 0,0
is at the most left and upper postion and the cell at 15,15 is at most right and down postion.

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
env.render()

total_reward = 0
for i in range(env.freeze * 100):
  print('.', end='')
  action = np.random.choice(list(env.actions_set))
  observation, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print('\nTotal Reward: {}'.format(total_reward))
```
## Images
Some camptures of env.render()
The red cross marks the position of the helicopter.

![A](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq0.svg)
**The forest at some time _t+0_**

![B](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq1.svg)
**The forest at some time _t+8_**

![C](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq2.svg)
**The forest at some time _t+16_**

![D](https://github.com/elbecerrasoto/gym-forest-fire/blob/master/pics/seq3.svg)
**The forest at some time _t+24_**
