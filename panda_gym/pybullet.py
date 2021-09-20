from contextlib import contextmanager
import time
import warnings

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data


class PyBullet:
    """Convenient class to use PyBullet physics engine.

    Args:
        render (bool, optional): Enable rendering. Defaults to False.
        n_substeps (int, optional): Number of sim substep when step() is
            called. Defaults to 20.
    """

    def __init__(self, render=False, n_substeps=20, background_color=(116, 160, 216)):
        self.background_color = [val / 255 for val in background_color]

        options = "--background_color_red={} \
                    --background_color_green={} \
                    --background_color_blue={}".format(
            *self.background_color
        )
        self.connection_mode = p.GUI if render else p.DIRECT
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self):
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self):
        """Close the simulation."""
        self.physics_client.disconnect()

    def render(
        self,
        mode="human",
        width=720,
        height=480,
        target_position=(0.0, 0.0, 0.0),
        distance=1.4,
        yaw=45,
        pitch=-30,
        roll=0,
    ):
        """Render.

        If mode is human, make the rendering real-time. All other arguments are
        unused. If mode is 'rgb_array', return an rgb_array of the scene.

        Args:
            mode (str, optional): 'human' of 'rgb_array'. If human, just sleep a
                few time to make the rendering real-time. Else, return an RGB
                array. Defaults to 'human'.
            width (int, optional): Image width. Defaults to 960.
            height (int, optional): Image height. Image height. Defaults to 720.
            target_position ((x, y, z), optional): Camera targetting this postion.
                Defaults to (0., 0., 0.).
            distance (float, optional): Distance of the camera. Defaults to 2.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -15.
            roll (float, optional): Rool of the camera. Defaults to 0.

        Returns:
            An RGB array if mode is 'rgb_array'.
        """
        if mode == "human":
            self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING)
            time.sleep(self.dt)  # wait to seems like real speed
        if mode == "rgb_array":
            if self.connection_mode == p.DIRECT:
                warnings.warn(
                    "The use of the render method is not recommended when the environment "
                    "has not been created with render=True. The rendering will probably be weird. "
                    "Prefer making the environment with option `render=True`. For example: "
                    "`env = gym.make('PandaReach-v1', render=True)`.",
                    UserWarning,
                )
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, px, depth, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            return px

    def get_base_position(self, body):
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (x, y, z): The cartesian position.
        """
        return self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[0]

    def get_base_orientation(self, body):
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (x, y, z, w): The orientation as quaternion.
        """
        return self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[1]

    def get_base_rotation(self, body):
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (rx, ry, rz): The rotation.
        """
        return self.physics_client.getEulerFromQuaternion(self.get_base_orientation(body))

    def get_base_velocity(self, body):
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (vx, vy, vz): The cartesian velocity.
        """
        return self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]

    def get_base_angular_velocity(self, body):
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            (wx, wy, wz): The angular velocity.
        """
        return self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]

    def get_link_position(self, body, link):
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (x, y, z): The cartesian position.
        """
        return self.physics_client.getLinkState(self._bodies_idx[body], link)[0]

    def get_link_orientation(self, body, link):
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (x, y, z, w): The orientation as quaternion.
        """
        return self.physics_client.getLinkState(self._bodies_idx[body], link)[1]

    def get_link_velocity(self, body, link):
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (vx, vy, vz): The cartesian velocity.
        """
        return self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]

    def get_link_angular_velocity(self, body, link):
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            (wx, wy, wz): The angular velocity.
        """
        return self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]

    def get_joint_angle(self, body, joint):
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def set_base_pose(self, body, position, orientation):
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (x, y, z): The target cartesian position.
            orientation (x, y, z, w): The target orientation as quaternion.
        """
        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_joint_angles(self, body, joints, angles):
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (List[int]): List of joint indices.
            angles (List[float]): List of target angles.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    def set_joint_angle(self, body, joint, angle):
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    def control_joints(
        self,
        body,
        joints,
        target_angles,
        forces,
    ):
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (List[int]): List of joint indices.
            target_angles (List[float]): List of target angles.
            forces (List[float]): Forces to apply.
        """
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def inverse_kinematics(self, body, ee_link, position, orientation):
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            ee_link (int): Link index of the end-effector.
            position (x, y, z): Desired position of the end-effector.
            orientation (x, y, z, w): Desired orientation of the end-effector.

        Returns:
            List[float]: The new joint state.
        """
        return self.physics_client.calculateInverseKinematics(
            bodyIndex=self._bodies_idx[body],
            endEffectorLinkIndex=ee_link,
            targetPosition=position,
            targetOrientation=orientation,
        )

    def place_visualizer(self, target_position, distance, yaw, pitch):
        """Orient the camera used for rendering.

        Args:
            target (x, y, z): Target position.
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self):
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name, **kwargs):
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def create_box(
        self,
        body_name,
        half_extents,
        mass,
        position,
        rgba_color,
        specular_color=[0, 0, 0, 0],
        ghost=False,
        friction=None,
    ):
        """Create a box.

        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            half_extents (x, y, z): Half size of the box in meters.
            mass (float): The mass in kg.
            position (x, y, z): The position of the box.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the box can collide. Defaults to False.
            friction (float, optionnal): The friction. If None, keep the pybullet default
                value. Defaults to None.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        return self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_cylinder(
        self,
        body_name,
        radius,
        height,
        mass,
        position,
        rgba_color,
        specular_color=[0, 0, 0, 0],
        ghost=False,
        friction=None,
    ):
        """Create a cylinder.

        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The radius in meter.
            mass (float): The mass in kg.
            position (x, y, z): The position of the sphere.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the sphere can collide. Defaults to False.
            friction (float, optionnal): The friction. If None, keep the pybullet default
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name,
        radius,
        mass,
        position,
        rgba_color,
        specular_color=[0, 0, 0, 0],
        ghost=False,
        friction=None,
    ):
        """Create a sphere.

        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (x, y, z): The position of the sphere.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the sphere can collide. Defaults to False.
            friction (float, optionnal): The friction. If None, keep the pybullet default
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name,
        geom_type,
        mass=0,
        position=(0, 0, 0),
        ghost=False,
        friction=None,
        visual_kwargs={},
        collision_kwargs={},
    ):
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (x, y, z): The position of the geom. Defaults to (0, 0, 0)
            ghost (bool, optional): Whether the geometry can collide. Defaults
                to False.
            friction (float, optionnal): The friction coef.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if friction is not None:
            self.physics_client.changeDynamics(
                bodyUniqueId=self._bodies_idx[body_name],
                linkIndex=-1,
                lateralFriction=friction,
            )

    def create_plane(self, z_offset):
        """Create a plane. (Actually it is a thin box)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=[3.0, 3.0, 0.01],
            mass=0,
            position=[0.0, 0.0, z_offset - 0.01],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.15, 0.15, 0.15, 1.0],
        )

    def create_table(self, length, width, height, x_offset=0):
        """Create a fixed table. Top is z=0, centered in y."""
        self.create_box(
            body_name="table",
            half_extents=[length / 2, width / 2, height / 2],
            mass=0,
            position=[x_offset, 0.0, -height / 2],
            specular_color=[0.0, 0.0, 0.0],
            rgba_color=[0.95, 0.95, 0.95, 1],
            friction=0.1,
        )

    def set_friction(self, body, link, friction):
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=friction,
        )
