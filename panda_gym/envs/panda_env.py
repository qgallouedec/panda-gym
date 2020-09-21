import numpy as np
import pybullet as p
from panda_gym.envs import robot_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PandaEnv(robot_env.RobotEnv):
    """Superclass for all Panda environments.
    """

    def __init__(self, model_path, n_substeps, gripper_extra_height,
                 block_gripper, has_object, target_in_the_air, target_offset,
                 obj_range, target_range, distance_threshold, initial_qpos,
                 reward_type, render):
        """Initializes a new Panda environment.

        Args:
            model_path (string): path to the environments json file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            render (bool): whether the rendering is enabled
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(PandaEnv, self).__init__(model_path=model_path,
                                       n_substeps=n_substeps,
                                       n_actions=4,
                                       initial_qpos=initial_qpos,
                                       render=render)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.reset_joint_states(obj_id=self.model['panda'],
                                    joint_indices=np.array([9, 10]),
                                    target_values=np.zeros(2))

    def _set_action(self, action):
        assert action.shape == (4, )
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        action = action.copy(
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        pos_ctrl *= 0.05  # limit maximum change in position
        gripper_ctrl *= 0.5

        # orientation of the effector
        # fixed rotation of the end effector, expressed as a quaternion
        target_orientation = [1, 0, 0, 0]

        # get the current position and the target position
        current_pos = p.getLinkState(self.model['panda'], 11)[0]
        gripper_pos = np.array([
            p.getJointState(self.model['panda'], 9)[0],
            p.getJointState(self.model['panda'], 10)[0]])
        target_position = current_pos + pos_ctrl
    
        # fingers
        assert gripper_ctrl.shape == (2,)
        gripper_target = gripper_pos + gripper_ctrl
        if self.block_gripper:
            gripper_target = np.zeros_like(gripper_target)


        # compute the new joint angles
        joint_poses = np.array(
            p.calculateInverseKinematics(
                self.model['panda'],
                endEffectorLinkIndex=11,
                targetPosition=target_position,
                targetOrientation=target_orientation)[0:7])

        # set the new position target
        joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        target_positions = np.concatenate((joint_poses, gripper_target))
        forces = np.array([87, 87, 87, 87, 12, 12, 12, 140, 140])
        p.setJointMotorControlArray(self.model['panda'],
                                    jointIndices=joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=target_positions,
                                    forces=forces)

    def _get_obs(self):
        dt = self.n_substeps * self.timestep
        # grip position and velocity
        grip_pos, _, _, _, _, _, grip_velp, _ = p.getLinkState(
            self.model['panda'], 11, computeLinkVelocity=True)
        # [x_gripper y_gripper z_gripper]    the position of the gripper
        grip_pos = np.array(grip_pos)
        # [vx_gripper vy_gripper vz_gripper] the velocity of the gripper
        grip_velp = np.array(grip_velp) * dt

        # position and velocity of every joints of the robot
        gripper_state = np.array([
            p.getJointState(self.model['panda'], 9)[0],
            p.getJointState(self.model['panda'], 10)[0]])

        gripper_vel = np.array([
            p.getJointState(self.model['panda'], 9)[1],
            p.getJointState(self.model['panda'], 10)[1]]) * dt

        if self.has_object:
            # position of the object
            object_pos, object_quaternion = p.getBasePositionAndOrientation(
                self.model['object'])
            # [x_obj, y_obj, z_obj] position of the object
            object_pos = np.array(object_pos)
            # [rx_obj, ry_obj, rz_obj] rotation under three axis of the object
            object_rot = np.array(p.getEulerFromQuaternion(object_quaternion))

            # velocities
            object_velp, object_velr = p.getBaseVelocity(self.model['object'])
            object_velr = np.array(object_velr) * dt
            object_velp = np.array(object_velp) * dt
            # [vx_obj , vy_obj , vz_obj ] position velocity of the object
            # [vrx_obj, vry_obj, vrz_obj] rotation velocity of the object

            # gripper state
            object_rel_pos = object_pos - grip_pos
            # ralative position between object and gripper

            object_velp -= grip_velp
            # the object velocity is taken with respect to the gripper velocity

        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(
                0)

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            gripper_state,
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            gripper_vel])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        # camera directed to the gripper position
        camera_target_position = p.getLinkState(self.model['panda'], 11)[0]
        p.resetDebugVisualizerCamera(cameraDistance=1.1,
                                     cameraYaw=48.,
                                     cameraPitch=-14.,
                                     cameraTargetPosition=camera_target_position)

    def _render_callback(self):
        # visualize the target
        obj_pos = np.concatenate((self.goal, np.array([0, 0, 0, 1])))
        self.reset_pos('target', obj_pos)

    def _reset_sim(self):
        p.restoreState(self.initial_state)
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]  # [x_obj, y_obj]
            while np.linalg.norm(object_xpos -
                                 self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range,
                                           self.obj_range, size=2)
            posObj = self.get_pos('object')

            assert posObj.shape == (7, )
            posObj[:2] = object_xpos

            self.reset_pos('object', posObj)
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        joint_ctrl = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.8])
        gripper_ctrl = np.array([0.0, 0.0])
        target_values = np.concatenate((joint_ctrl, gripper_ctrl))

        self.reset_joint_states(obj_id=self.model['panda'],
                                joint_indices=joint_indices,
                                target_values=target_values)

        # compute the initial position with extra_height
        target_position = np.array(p.getLinkState(self.model['panda'], 11)[0])
        target_position[2] += self.gripper_extra_height
        target_orientation = np.array([1., 0., 0., 0.])

        # compute the new joint angles
        jointPoses = np.array(
            p.calculateInverseKinematics(
                self.model['panda'],
                endEffectorLinkIndex=11,
                targetPosition=target_position,
                targetOrientation=target_orientation)[0:7])

        # set the new position target
        target_values = np.concatenate((jointPoses, gripper_ctrl))

        self.reset_joint_states(obj_id=self.model['panda'],
                                joint_indices=joint_indices,
                                target_values=target_values)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = np.array(
            p.getLinkState(self.model['panda'], 11)[0]).copy()

        if self.has_object:
            # move the object if a new position is given in initial_qpos
            if initial_qpos.get('object') is not None:
                obj_target_pos = initial_qpos.get('object')
                self.reset_pos('object', obj_target_pos)
            else:
                obj_target_pos = self.get_pos('object')

            self.height_offset = obj_target_pos[2]

    def render(self, mode='human', width=960, height=720):
        # camera directed to the gripper position
        target_position = [1.3, 0.75, 0.4]
        distance = 1.4
        yaw = 48.
        pitch = -14.
        return super(PandaEnv, self).render(mode, width, height, target_position,
                                            distance, yaw, pitch)
