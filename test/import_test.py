# Basic package test
import os

import gym
import panda_gym

def main():
    """
    Tests importing of gym envs
    """
    try:
        spec = gym.spec('PandaReach-v0')
    except:
        return False
    return True


def test_import():
    assert main()


if __name__ == '__main__':
    main()
