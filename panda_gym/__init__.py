from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    register(
        id="PandaReach{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaPush{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPushEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaSlide{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaSlideEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaPickAndPlace{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaStack{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaStackEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )
