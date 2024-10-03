from typing import Optional

import numpy as np
import pathlib as p
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

assets_path = str(p.Path(__file__).parent.parent.parent) + "/assets"

class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        ee_orientation = np.array(self.get_ee_orientation())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, ee_orientation, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity, ee_orientation))
        return observation
    
    
    def set_obs(self, observation) -> np.ndarray:
        ee_position = observation[0:3]
        ee_velocity = observation[3:6]
        ee_orientation = observation[6:10]
        fingers_width = observation[10] if not self.block_gripper else 0

        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=ee_position, orientation=ee_orientation
        )
        target_arm_angles[-2] = fingers_width / 2
        target_arm_angles[-1] = fingers_width / 2
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices[-9:], angles=target_arm_angles[-9:])

    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as (x, y, z)"""
        return self.get_link_orientation(self.ee_link)



    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)







class MobilePanda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 4 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        n_action += 2 # left and right wheels
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="mobile_panda",
            file_name=assets_path+"/mobile_panda_desc/mobile_panda.urdf",
            base_position=self.base_position,
            action_space=action_space,

            joint_indices=np.array([2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 18, 19 ]),
            joint_forces=np.array([70.0, 70.0, 70.0, 70.0, 87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
            fixed_base = False,
        )

        self.fingers_indices = np.array([18, 19])
        self.neutral_joint_values = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.41, 0.00, -1.95, 0.00, 2.76, 0.79, 0.00, 0.00])

        self.ee_link = 20
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        wheels = action[:2].copy()
        wheels = self.throttle_to_wheel_speed(5*wheels[0],5*wheels[1])
        action = action[2:].copy()
        if self.control_type == "ee":
            ee_displacement = action[:4]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_values = np.concatenate((wheels,target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_robot(targets=target_values)

    def control_robot(self,targets):
        self.sim.control_velocities(
        body=self.body_name,
        joints=self.joint_indices[:4],
        target_velocities=targets[:4],
        forces=self.joint_forces[:4],
        )
        self.sim.control_joints(
        body=self.body_name,
        joints=self.joint_indices[4:],
        target_angles=targets[4:],
        forces=self.joint_forces[4:],
        )

    def ee_displacement_to_target_arm_angles(self, displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement

        ee_orientation = self.get_ee_orientation()
        target_ee_orientation = np.array(self.sim.physics_client.getEulerFromQuaternion(ee_orientation))
        target_ee_orientation[0] = np.pi
        target_ee_orientation[1] =0
        target_ee_orientation[2] += displacement[3] * 0.03
        target_ee_orientation = self.sim.physics_client.getQuaternionFromEuler(target_ee_orientation)

        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=target_ee_orientation
        )
        target_arm_angles = target_arm_angles[4:11]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint) for joint in self.joint_indices[4:-2]])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles
    
    def throttle_to_wheel_speed(self,t1,t2):
        v1 = self.get_joint_velocity(joint=0)
        v2 = self.get_joint_velocity(joint=1)
        return np.array([v1+t1,v2+t2,v1+t1,v2+t2])

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_orientation = np.array(self.get_ee_orientation())
        ee_velocity = np.array(self.get_ee_velocity())
        bp = self.sim.get_base_position(self.body_name)
        bo = self.sim.get_base_orientation(self.body_name)
        bv = self.sim.get_base_angular_velocity(self.body_name)
        bav = self.sim.get_base_velocity(self.body_name)
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, ee_orientation, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity, ee_orientation))
        observation = np.concatenate((bp,bo,bv,bav,observation))
        return observation
    
    def set_obs(self, observation) -> np.ndarray:
        bp = observation[:3]
        bo = observation[3:7]
        bv = observation[7:10]
        bav = observation[10:13]
        ee_position = observation[13:16]
        ee_velocity = observation[16:19]
        ee_orientation = observation[19:23]
        fingers_width = observation[23] if not self.block_gripper else 0

        self.sim.set_base_pose(self.body_name, position=bp, orientation=bo)
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=ee_position, orientation=ee_orientation
        )
        target_arm_angles[-2] = fingers_width / 2
        target_arm_angles[-1] = fingers_width / 2
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices[-9:], angles=target_arm_angles[-9:])

    def reset(self) -> None:
        self.set_joint_neutral()
        rand = 0.1*np.random.uniform(size=3)
        rand[2]=0
        self.sim.set_base_pose(self.body_name, position=self.base_position, orientation=np.array([0.0, 0.0, 2*np.pi*np.random.uniform()]))
        self.sim.set_base_pose(self.body_name, position=self.base_position, orientation=np.array([0.0, 0.0, 0]))



    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)
    
    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as (x, y, z)"""
        return self.get_link_orientation(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
