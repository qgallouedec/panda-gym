import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

import time

import json

GEOM_TYPE = {
    "box": p.GEOM_BOX,
    "sphere": p.GEOM_SPHERE,
    "cylinder": p.GEOM_CYLINDER
}


def load_model_from_path(path):
    with open(path, 'r') as file:
        obj_dicts = json.load(file)

    model = {}
    p.resetSimulation()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    for obj_dict in obj_dicts:
        obj_name, obj_type, obj_args = obj_dict['name'], obj_dict['type'], obj_dict['args']

        if obj_type == 'urdf':
            model[obj_name] = p.loadURDF(**obj_args)

        elif obj_type == 'geom':
            shapeType = GEOM_TYPE[obj_args['type']]
            if obj_args['ghost']:
                baseCollisionShapeIndex = -1
            else:
                baseCollisionShapeIndex = p.createCollisionShape(
                    shapeType, **obj_args['shape'])

            try:
                obj_args['shape']['length'] = obj_args['shape'].pop('height')
            except KeyError:
                pass

            baseVisualShapeIndex = p.createVisualShape(
                shapeType, **obj_args['shape'], **obj_args['visual'])

            model[obj_name] = p.createMultiBody(
                baseVisualShapeIndex=baseVisualShapeIndex,
                baseCollisionShapeIndex=baseCollisionShapeIndex,
                **obj_args['body'])

    # # re-enable visualisation
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return model


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, render):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(
                __file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        if render:
            options = '--background_color_red={} --background_color_green={} --background_color_blue={}'.format(116./255., 220./255., 146./255.)
            p.connect(p.GUI, options=options)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        else:
            p.connect(p.DIRECT)

        self.model = load_model_from_path(fullpath)

        self.n_substeps = n_substeps
        self.timestep = 1./500
        p.setTimeStep(self.timestep)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self._viewer_setup()

        self.initial_state = copy.deepcopy(p.saveState())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1.,
                                       shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.timestep * self.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._sim_step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        return obs, reward, done, info

    def _sim_step(self):
        for _ in range(self.n_substeps):
            p.stepSimulation()

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        p.disconnect()

    def render(self, mode='human', width=960, height=720, target_position=[0., 0., 0.], distance=2., yaw=45., pitch=-15., roll=0.):
        self._render_callback()
        if mode == 'human':
            time.sleep(self.dt) # wait to seems like real speed
        if mode == 'rgb_array':
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position, distance=distance,
                yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=2)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height,
                nearVal=0.1, farVal=100.0)
            (_, _, px, depth, _) = p.getCameraImage(
                width=width, height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                # renderer=p.ER_TINY_RENDERER,
                lightDirection=[1., -0.9, 2.],
                lightColor=[1., 1., 1.],
                lightDistance=1,
                lightAmbientCoeff=0.3,
                lightDiffuseCoeff=0.7,
                lightSpecularCoeff=0.7
                )
            
            # configure background color
            for ix in range(len(px)):
                for iy in range(len(px[ix])):
                    if depth[ix][iy] > 0.99:
                        px[ix][iy][:] = [116., 220., 146., 255.]

            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        p.restoreState(stateId=self.initial_state)
        return True

    def get_pos(self, body_id):
        """Return the position and the orientation of an object"""
        position_obj, orientation_obj = p.getBasePositionAndOrientation(
            bodyUniqueId=self.model[body_id])
        out = np.concatenate((
            np.array(position_obj),
            np.array(orientation_obj)
        ))
        return out

    def reset_pos(self, obj_id, obj_pos_orn):
        """Reset the position and orientation of an object

        obj_pos_orn is a numpy.ndarray where the 3 first coordinates
        are the position and 4 others are the orientation (as a quaternion)
        """
        assert obj_pos_orn.shape == (7,)
        obj_pos, obj_orn = obj_pos_orn[:3], obj_pos_orn[3:]
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self.model[obj_id],
            posObj=obj_pos,
            ornObj=obj_orn)

    def reset_joint_states(self, obj_id, joint_indices, target_values):
        """Reset the joint values
        joint_indices is a numpy.ndarray of indices of joints 
        target_values is a numpy.ndarray of the corresponding values
        """
        assert joint_indices.shape == target_values.shape
        for joint_index, target_value in zip(joint_indices, target_values):
            self.reset_joint_state(
                obj_id=obj_id,
                joint_index=joint_index,
                target_value=target_value)

    def reset_joint_state(self, obj_id, joint_index, target_value):
        """Reset the joint values
        joint_indices is a numpy.ndarray of indices of joints 
        target_values is a numpy.ndarray of the corresponding values
        """
        p.resetJointState(
            bodyUniqueId=obj_id,
            jointIndex=joint_index,
            targetValue=target_value)

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
