.. _flip:

Flip
====

The task consists in flipping a cube to a randomly generated target orientation.

Goal
----

The goal is composed of 4 coordinates, corresponding to the target rotation as quaternion. It is represented as a transparent cube in the rendering. The task is achieved when the geodesic distance between the object orientation and the target orientation is below 0.2.

Task observation
----------------

The observation for the task is composed of 13 coordinates:

- the object position (3 coordinates),
- the object rotation as quaternion (4 coordinates),
- the object velocity (3 coordinates) and
- the object angular velocity (3 coordinates)




