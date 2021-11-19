.. _train_with_sb3:

Train with stable-baselines3
============================

You can train the environments with any OpenAI/gym compatible library. In this documentation we explain how to use one of them: `stable-baselines3 (SB3) <https://stable-baselines3.readthedocs.io/en/master/index.html>`_.

Install SB3
-----------

To install SB3, follow the instructions from its documentation `Install stable-baselines3 <https://stable-baselines3.readthedocs.io/en/master/guide/install.html>`_.


Alternatively, you can install ``panda-gym`` and SB3 directly with a single command: 

.. code-block:: bash

    pip install panda-gym[extra]

.. warning::
    If you use `zsh` terminal, the syntax is ``pip install 'panda-gym[extra]'``

Train
-----

Now that SB3 is installed, you can run the following code to train an agent. You can use every algorithm compatible with ``Box`` action space, see `stable-baselines3/RL Algorithm <https://stable-baselines3.readthedocs.io/en/master/guide/algos.html>`_). In the following example, a DDPG agent is trained to solve th Reach task.

.. code-block:: python

    import gym
    import panda_gym
    from stable_baselines3 import DDPG

    env = gym.make("PandaReach-v2")
    model = DDPG(policy="MultiInputPolicy", env=env)
    model.train(30000)

.. note::

    Here we provide the canonical code for training with SB3. For any information on the setting of hyperparameters, verbosity, saving the model, ... please read the `SB3 documentation <https://stable-baselines3.readthedocs.io/en/master/index.html>`_. 
 

Bonus: Train with RL Baselines3 Zoo
-----------------------------------

`RL Baselines3 Zoo <https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html>`_ is the training framework associated with SB3.
It provides scripts for training, evaluating agents, setting hyperparameters, plotting results and recording video. It also contains already optimized hypermeters, including for some ``panda-gym`` environments.

.. warning::
    The current version of RL Baselines3 Zoo provides hyperparameters for version 1 of ``panda-gym``, but not for version 2. Before training with RL Baselines3 Zoo, you will have to set your own hyperparameters by editing ``hyperparameters/<ALGO>.yml``. For more information, please read the `README of RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo#readme>`_.

Train
~~~~~

To use it, follow the `instructions for its installation <https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html#installation>`_, then use the following command.

.. code-block:: bash

    python train.py --algo <ALGO> --env <ENV>

For example, to train an agent with TQC on ``PandaPickAndPlace-v2``:

.. code-block:: bash

    python train.py --algo tqc --env PandaPickAndPlace-v2

Enjoy
~~~~~

To visualize the trained agent, follow the `instructions <https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html#enjoy-a-trained-agent>`_ in the SB3 documentation. It is necessary to add ``--env-kwargs render:True`` when running the enjoy script.

.. code-block:: bash 

    python enjoy.py --algo <ALGO> --env <ENV> --folder <TRAIN_AGENT_FOLDER> --env-kwargs render:True