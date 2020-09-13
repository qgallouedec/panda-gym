from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='PandaSlide{}-v0'.format(suffix),
        entry_point='panda_gym.envs:PandaSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='PandaPickAndPlace{}-v0'.format(suffix),
        entry_point='panda_gym.envs:PandaPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='PandaReach{}-v0'.format(suffix),
        entry_point='panda_gym.envs:PandaReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='PandaPush{}-v0'.format(suffix),
        entry_point='panda_gym.envs:PandaPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
