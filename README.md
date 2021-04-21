# panda-gym

OpenaAI Gym Franka Emika Panda robot environment based on PyBullet.

[![PyPI version](https://img.shields.io/pypi/v/panda-gym.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/panda-gym/)
[![Downloads](https://pepy.tech/badge/panda-gym)](https://pepy.tech/project/panda-gym)
[![GitHub](https://img.shields.io/github/license/qgallouedec/panda-gym.svg)](LICENSE.txt)
[![build](https://github.com/qgallouedec/panda-gym/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/qgallouedec/panda-gym/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/qgallouedec/panda-gym/branch/master/graph/badge.svg?token=pv0VdsXByP)](https://codecov.io/gh/qgallouedec/panda-gym)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

### Using PyPI

```bash
pip install panda-gym
```

### From source

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
| ![PandaReach-v1](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/reach.gif) |         ![PandaPush-v1](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/push.gif)         |
|         `PandaSlide-v1`          |             `PandaPickAndPlace-v1`             |
| ![PandaSlide-v1](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/slide.gif) | ![PandaPickAndPlace-v1](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/pickandplace.gif) |
|         `PandaStack-v1`          |                                                |
| ![PandaStack-v1](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/stack.gif) |                                                |

## Citation

Cite as

```
@misc{panda_gym,
  author = {Gallou\'{e}dec, Quentin},
  title = {Multi-goal reinforcement learning environments for Franka Emika Panda robot},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/qgallouedec/panda-gym}},
}
```

Environments are widely inspired from [OpenAI Fetch environments](https://openai.com/blog/ingredients-for-robotics-research/). 
