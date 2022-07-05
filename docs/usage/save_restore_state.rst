.. _save_restore_states:

Save and Restore States
==============

It is possible to save a state of the entire simulation environment. This is useful if your application requires lookahead search. Below is an example of a greedy random search.

.. code-block:: python

    import gym
    import panda_gym

    env = gym.make("PandaReachDense-v2", render=True)
    obs = env.reset()

    while True:
        state_id = env.save_state()
        best_action = None
        rew = best_rew = env.task.compute_reward(
            obs["achieved_goal"], obs["desired_goal"], None) 

        while rew <= best_rew:
            env.restore_state(state_id)
            a = env.action_space.sample()
            _, rew, _, _ = env.step(a)

        env.restore_state(state_id)
        obs, _, _, _ = env.step(a)
        env.remove_state(state_id)

    env.close()