import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class MyRobot(PyBulletRobot):
    """My robot"""

    def __init__(self, sim) -> None:
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="my_robot",  # choose the name you want
            file_name="docs/files/my_robot.urdf",  # the path of the URDF file
            base_position=np.zeros(3),  # the position of the base
            action_space=action_space,
            joint_indices=np.array([0]),  # list of the indices, as defined in the URDF
            joint_forces=np.array([1.0]),  # force applied when robot is controled (Nm)
        )

    def set_action(self, action):
        self.control_joints(target_angles=action)

    def get_obs(self):
        return self.get_joint_angle(joint=0)

    def reset(self) -> None:
        # Sets the neutral position of the joint. Here, at angle=0
        neutral_angle = np.array([0.0])
        self.set_joint_angles(angles=neutral_angle)


if __name__ == "__main__":
    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render=True)
    robot = MyRobot(sim)

    for _ in range(50):
        robot.set_action(np.array([1.0]))
        sim.step()
        sim.render()
