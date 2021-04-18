import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot


class Panda(PyBulletRobot):
    """Panda in PyBullet.

    Args:
        sim (Any): Simulation engine.
        block_gripper (bool, optional): Whether the gripper is blocked.
            Defaults to False.
        base_position ((x, y, z), optionnal): Position of the base base of the robot.
        fingers_friction (float, optional): Lateral friction of the fingers. Defaults to 1.0.
    """

    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    FINGERS_INDICES = [9, 10]
    NEUTRAL_JOINT_VALUES = [0.00, 0.41, 0.00, -1.85, -0.00, 2.26, 0.79, 0, 0]
    JOINT_FORCES = [87, 87, 87, 87, 12, 120, 120, 170, 170]

    def __init__(
        self, sim, block_gripper=False, base_position=[0, 0, 0], fingers_friction=1.0
    ):
        self.block_gripper = block_gripper
        n_action = 3 if self.block_gripper else 4
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_action,))
        self.ee_link = 11
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
        )
        self.sim.set_friction(self.body_name, self.FINGERS_INDICES[0], fingers_friction)
        self.sim.set_friction(self.body_name, self.FINGERS_INDICES[1], fingers_friction)

    def set_action(self, action):
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ee_ctrl = action[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_ctrl
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = max(0, target_ee_position[2])

        # compute the new joint angles
        target_angles = self._inverse_kinematics(
            position=target_ee_position, orientation=[1, 0, 0, 0]
        )

        if not self.block_gripper:
            fingers_ctrl = action[3] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl
            target_angles[-2:] = [target_fingers_width / 2, target_fingers_width / 2]

        self.control_joints(target_angles=target_angles)

    def get_obs(self):
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            obs = np.concatenate((ee_position, ee_velocity))
        return obs

    def reset(self):
        self.set_joint_neutral()

    def get_fingers_width(self):
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.FINGERS_INDICES[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.FINGERS_INDICES[1])
        return finger1 + finger2

    def get_ee_position(self):
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self):
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def _inverse_kinematics(self, position, orientation):
        """Compute the inverse kinematics and return the new joint values. The last two
        coordinates (fingers) are [0, 0].

        Args:
            position (x, y, z): Desired position of the end-effector.
            orientation (x, y, z, w): Desired orientation of the end-effector.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(
            self.body_name, ee_link=11, position=position, orientation=orientation
        )
        # Replace the fingers coef by [0, 0]
        inverse_kinematics = list(inverse_kinematics[0:7]) + [0, 0]
        return inverse_kinematics

    def set_joint_neutral(self):
        """Set the robot to its neutral pose."""
        self.set_joint_values(self.NEUTRAL_JOINT_VALUES)

    def set_ee_position(self, position):
        """Set the end-effector position. Can induce collisions.

        Warning:
            Make sure that the position does not induce collision.

        Args:
            position (x, y, z): Desired position of the gripper.
        """
        # compute the new joint angles
        angles = self._inverse_kinematics(
            position=position, orientation=[1.0, 0.0, 0.0, 0.0]
        )
        self.set_joint_values(angles=angles)

    def set_joint_values(self, angles):
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(
            self.body_name, joints=self.JOINT_INDICES, angles=angles
        )
