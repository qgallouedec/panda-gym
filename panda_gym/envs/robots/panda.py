from typing import Any, List

import numpy as np
from gym import spaces

from panda_gym.envs.core import RobotEnv


class PandaEnv(RobotEnv):
    """Superclass for all Panda environments.

    Args:
        sim (Any): Simulation engine.
        block_gripper (bool, optional): Whetther the gripper is blocked.
            Defaults to False.
    """

    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    NEUTRAL_JOINT_VALUES = [0.00, 0.41, 0.00, -1.85, -0.00, 2.26, 0.79, 0, 0]

    def __init__(self, sim: Any, block_gripper: bool = False, base_position=[0,0,0]):
        self.action_space = spaces.Box(-1., 1., shape=(4,))
        self.block_gripper = block_gripper

        super().__init__(sim, body_name='panda', ee_link=11,
                         file_name="franka_panda/panda.urdf",
                         base_position=base_position)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        return (self.sim.get_joint_angle(self.body_name, 9)
                + self.sim.get_joint_angle(self.body_name, 10))

    def inverse_kinematics(
            self, position, orientation) -> List:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            position (x, y, z): Desired position of the end-effector.
            orientation (x, y, z, w): Desired orientation of the end-effector.

        Returns:
            List: Joint values.
        """
        return list(self.sim.inverse_kinematics(
            self.body_name, ee_link=11, position=position,
            orientation=orientation)[0:7]) + [0, 0]  # [0, 0] for the fingers

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_values(self.NEUTRAL_JOINT_VALUES)

    def set_ee_position(self, position) -> None:
        """Set the end-effector position. Can induce collisions.

        Warning:
            Make sure that the position does not induce collision.

        Args:
            position (x, y, z): Desired position of the gripper.
        """
        # compute the new joint angles
        angles = self.inverse_kinematics(position=position,
                                         orientation=[1., 0., 0., 0.])
        self.set_joint_values(angles=angles)

    def set_joint_values(self, angles: List) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (List): Joint angles.
        """
        self.sim.set_joint_angles(
            self.body_name, joints=self.JOINT_INDICES, angles=angles)

    def control_joints(self, target_angles: List) -> None:
        """Control the joint angles.

        Args:
            target_angles (List): The target angles of the joints.
        """
        forces = np.array([87, 87, 87, 87, 12, 120, 120, 20, 20])
        self.sim.control_joints(
            body=self.body_name, joints=self.JOINT_INDICES,
            target_angles=target_angles, forces=forces)

    def _set_action(self, action):
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ee_ctrl, fingers_ctrl = action[:3], action[3]

        # limit maximum change in position
        ee_ctrl *= 0.05
        fingers_ctrl *= 0.5

        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_ctrl

        # compute the new joint angles
        target_angles = self.inverse_kinematics(position=target_ee_position,
                                                orientation=[1, 0, 0, 0])

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles[-2:] = [target_fingers_width/2, target_fingers_width/2]
        self.control_joints(target_angles=target_angles)

        if self.block_gripper:
            self.sim.set_joint_angles(
                self.body_name, joints=[9, 10], angles=[0, 0])

    def _env_setup(self):
        super()._env_setup()
        self.set_joint_neutral()

    # def render(self, mode='human', width=960, height=720):
    #     # camera directed to the gripper position
    #     target_position = [0.0, 0.0, 0.0]
    #     distance = 0.6
    #     yaw = 90. # 48
    #     pitch = -55. #-14
    #     return self.sim.render(mode, width, height, target_position,
    #                            distance, yaw, pitch)
