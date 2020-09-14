# panda-gym

OpenaAI Gym Franka Emika Panda robot environment based on PyBullet.

## Installation

Using PyPI:

    pip install panda-gym

From source:

    git clone https://github.com/quenting44/panda-gym.git
    pip install -e panda-gym

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

Environnement are widely inspired from [OpenAI Fetch environnements](https://openai.com/blog/ingredients-for-robotics-research/).

`PandaReach-v0`: Panda has to move its end-effector to the desired goal position.
![](https://raw.githubusercontent.com/quenting44/panda-gym/master/docs/Reach.png)

`PandaSlide-v0`: Panda has to hit a puck across a long table such that it slides and comes to rest on the desired goal.
![](https://raw.githubusercontent.com/quenting44/panda-gym/master/docs/Slide.png)

`PandaPush-v0`: Panda has to move a box by pushing it until it reaches a desired goal position.
![](https://raw.githubusercontent.com/quenting44/panda-gym/master/docs/Push.png)

`PandaPickAndPlace-v0`: Panda has to pick up a box from a table using its gripper and move it to a desired goal above the table.
![](https://raw.githubusercontent.com/quenting44/panda-gym/master/docs/PickAndPlace.png)