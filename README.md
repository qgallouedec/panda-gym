# panda-gym

**Under development**

[![PyPI version](https://img.shields.io/pypi/v/panda-gym.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/panda-gym/)
[![Downloads](https://pepy.tech/badge/panda-gym)](https://pepy.tech/project/panda-gym)
[![GitHub](https://img.shields.io/github/license/qgallouedec/panda-gym.svg)](LICENSE.txt)
[![Actions Status](https://github.com/qgallouedec/panda-gym/workflows/build/badge.svg)](https://github.com/qgallouedec/panda-gym/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

OpenaAI Gym Franka Emika Panda robot environment based on PyBullet.

![gif_demo](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/demo.gif)

## Installation

Using PyPI:

```bash
pip install panda-gym
```

From source:

```bash
git clone https://github.com/qgallouedec/panda-gym.git
pip install -e panda-gym
```

## Usage

```python
import gym
import panda_gym

env = gym.make('PandaReach-v1', render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)

env.close()
```

## Environments

|                                  |                                                |
| :------------------------------: | :--------------------------------------------: |
|         `PandaReach-v1`          |                 `PandaPush-v1`                 |
| ![PandaReach-v1](docs/reach.gif) |         ![PandaPush-v1](docs/push.gif)         |
|         `PandaSlide-v1`          |             `PandaPickAndPlace-v1`             |
| ![PandaSlide-v1](docs/slide.gif) | ![PandaPickAndPlace-v1](docs/pickandplace.gif) |
|         `PandaStack-v1`          |                                                |
| ![PandaStack-v1](docs/stack.gif) |                                                |

Environments are widely inspired from [OpenAI Fetch environments](https://openai.com/blog/ingredients-for-robotics-research/). 