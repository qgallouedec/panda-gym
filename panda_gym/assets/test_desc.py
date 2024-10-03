import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
p.connect(p.GUI)

# Set the search path to find the URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the combined URDF
robot_id = p.loadURDF("mobile_panda_desc/mobile_panda.urdf", basePosition=[0, 0, 0], useFixedBase=False)

# Optionally, you can add a plane for the robot to move on
plane_id = p.loadURDF("plane.urdf")

# Set gravity
p.setGravity(0, 0, -9.8)

import numpy as np
joint_indices=np.array([2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 20, 21 ])
joint_indices=np.array([2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 17, 18 ])
joint_indices[4:]+=1
#0f 1f 2cont 3cont 4cont 5cont 6f 7f 8f 9f 10f 11rot 12rot 13rot 14rot 15rot 16rot 17rot 18f 19f 20prism 21prism 22f
joint_forces=np.array([100.0, 100.0, 100.0, 100.0, 87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])

ingers_indices = np.array([20, 21])
neutral_joint_values = np.array([0.00, 1.00, 0.00, 0.00, 1.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
neutral_joint =        np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 0.00, 1.00])
target_velocities = np.array([0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.0, 0.00, 0.00, 0.00, 0.00, 0.00])

neutral_joint = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.41, 0.00, -1.95, 0.00, 2.76, 0.79, 0.00, 0.00])
neutral_joint = np.array([ 0,0,0, 0,-0.0581066,0.66762043,0.0586549 , -1.97301356 , 0.0629228  , 2.75354837 , 0.73150214 , 0,0       ])
# p.setJointMotorControlArray(
#             robot_id,
#             jointIndices=joint_indices[:4],
#             controlMode=p.VELOCITY_CONTROL,
#             targetVelocities = target_velocities[:4],
#             forces=joint_forces[:4],
#         )

p.setJointMotorControlArray(
            robot_id,
            jointIndices=joint_indices[4:],
            controlMode=p.POSITION_CONTROL,
            targetPositions = neutral_joint[4:],
            forces=joint_forces[4:],
        )

# Run the simulation
while True:
    p.stepSimulation()
    time.sleep(1./240.)




# import pybullet as p
# import pybullet_data
# import time

# # Connect to PyBullet
# p.connect(p.GUI)

# # Set the search path to find the URDF files
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Load the combined URDF
# robot_id = p.loadURDF("mobile_panda_desc/mobile_panda.urdf", basePosition=[0, 0, 0], useFixedBase=False)

# # Optionally, you can add a plane for the robot to move on
# plane_id = p.loadURDF("plane.urdf")

# # Set gravity
# p.setGravity(0, 0, -9.8)

# import numpy as np

# joint_indices = np.array([2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 17, 18])
# joint_indices[4:]+=1
# joint_forces = np.array([100.0, 100.0, 100.0, 100.0, 87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])

# neutral_joint = np.array([0.00, 0.00, 0.00, 0.00, 0.00, -1.21, 0.00, -2.35, 0.00, 1.26, 0.79, 0.00, 0.00])

# # Initialize joint positions
# joint_positions = neutral_joint.copy()

# # Set initial joint positions
# p.setJointMotorControlArray(
#     robot_id,
#     jointIndices=joint_indices,
#     controlMode=p.POSITION_CONTROL,
#     targetPositions=joint_positions,
#     forces=joint_forces,
# )

# def update_joint_positions():
#     p.setJointMotorControlArray(
#         robot_id,
#         jointIndices=joint_indices,
#         controlMode=p.POSITION_CONTROL,
#         targetPositions=joint_positions,
#         forces=joint_forces,
#     )

# # Run the simulation
# while True:
#     # Capture user input for joint control
#     try:
#         joint_num = int(input("Enter joint number (0-12): "))
#         joint_val = float(input("Enter joint value: "))
        
#         if 0 <= joint_num < len(joint_positions):
#             joint_positions[joint_num] = joint_val
#             p.resetJointState(robot_id,joint_indices[joint_num],joint_val)
#         else:
#             print("Invalid joint number.")
#     except ValueError:
#         print("Invalid input. Please enter numeric values.")
    
#     p.stepSimulation()
#     time.sleep(1./240.)
