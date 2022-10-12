.. _save_restore_states:

Save and Restore States
==============

It is possible to save a state of the entire simulation environment. This is useful if your application requires lookahead search. Below is an example of a greedy random search.

.. code-block:: python

    import gymnasium as gym
    import panda_gym

    env = gym.make("PandaReachDense-v2", render=True)
    observation, _ = env.reset()

    for _ in range(1000):
        state_id = env.save_state()
        reward = best_reward = env.task.compute_reward(
            observation["achieved_goal"], observation["desired_goal"], None) 

        while reward <= best_reward:
            env.restore_state(state_id)
            action = env.action_space.sample()
            observation, reward, _, _, _ = env.step(action)

        env.restore_state(state_id)
        observation, reward, terminated, truncated, info = env.step(action)
        env.remove_state(state_id)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()