# panda-gym

[![PyPI version](https://img.shields.io/pypi/v/panda-gym.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/panda-gym/)
[![PyPI downloads](https://static.pepy.tech/badge/panda-gym)](https://pypistats.org/packages/panda-gym)
[![GitHub](https://img.shields.io/github/license/qgallouedec/panda-gym.svg)](LICENSE.txt)
[![Actions Status](https://github.com/qgallouedec/panda-gym/workflows/build/badge.svg)](https://github.com/qgallouedec/panda-gym/actions)

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

env = gym.make('PandaReach-v0', render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)

env.close()
```

## Environments

Following environments are widely inspired from [OpenAI Fetch environments](https://openai.com/blog/ingredients-for-robotics-research/). Video [here](https://youtu.be/TbISn3yu0CM). See my [blog post](https://qgallouedec.github.io/posts/2021/02/openai-environment-for-franka-emika-panda-robot/).

`PandaReach-v0`: Panda has to move its end-effector to the desired goal position.
![PandaReach-v0](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/reach.gif)

`PandaSlide-v0`: Panda has to hit a puck across a long table such that it slides and comes to rest on the desired goal.
![PandaSlide-v0](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/slide.gif)

`PandaPush-v0`: Panda has to move a box by pushing it until it reaches a desired goal position.
![PandaPush-v0](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/push.gif)

`PandaPickAndPlace-v0`: Panda has to pick up a box from a table using its gripper and move it to a desired goal above the table.
![PandaPickAndPlace-v0](https://raw.githubusercontent.com/qgallouedec/panda-gym/master/docs/pickandplace.gif)
