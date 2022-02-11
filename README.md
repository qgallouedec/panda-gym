# panda-gym

Set of OpenAI/gym robotic environments based on PyBullet physics engine.

[![PyPI version](https://img.shields.io/pypi/v/panda-gym.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/panda-gym/)
[![Downloads](https://pepy.tech/badge/panda-gym)](https://pepy.tech/project/panda-gym)
[![GitHub](https://img.shields.io/github/license/qgallouedec/panda-gym.svg)](LICENSE.txt)
[![build](https://github.com/qgallouedec/panda-gym/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/qgallouedec/panda-gym/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/qgallouedec/panda-gym/branch/master/graph/badge.svg?token=pv0VdsXByP)](https://codecov.io/gh/qgallouedec/panda-gym)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](https://img.shields.io/badge/cs.LG-arXiv%3A2106.13687-B31B1B.svg)](https://arxiv.org/abs/2106.13687)

## Documentation

Check out the [documentation](https://panda-gym.readthedocs.io/en/latest/).

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

env = gym.make('PandaReach-v2', render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)

env.close()
```

You can also [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qgallouedec/panda-gym/blob/master/examples/PickAndPlace.ipynb)

## Environments

|                                  |                                                |
| :------------------------------: | :--------------------------------------------: |
|         `PandaReach-v2`          |                 `PandaPush-v2`                 |
| ![PandaReach-v2](docs/_static/img/reach.png) |         ![PandaPush-v2](docs/_static/img/push.png)         |
|         `PandaSlide-v2`          |             `PandaPickAndPlace-v2`             |
| ![PandaSlide-v2](docs/_static/img/slide.png) | ![PandaPickAndPlace-v2](docs/_static/img/pickandplace.png) |
|         `PandaStack-v2`          |              `PandaFlip-v2`                    |
| ![PandaStack-v2](docs/_static/img/stack.png) | ![PandaFlip-v2](docs/_static/img/flip.png) |

## Baselines results

Baselines results and pre-trained agents available in [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

## Citation

Cite as

```bib
@article{gallouedec2021pandagym,
  title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
  year         = 2021,
  journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}
```

Environments are widely inspired from [OpenAI Fetch environments](https://openai.com/blog/ingredients-for-robotics-research/). 
