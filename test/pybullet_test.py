import pytest


def test_import():
    from panda_gym.pybullet import PyBullet


def test_construct():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()


def test_close():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.close()


def test_step():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.step()
    pybullet.close()


def test_dt():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    assert pybullet.dt == 0.04
    pybullet.close()


def test_create_box():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    pybullet.close()


def test_get_base_position():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    base_position = pybullet.get_base_position("my_box")
    pybullet.close()
    assert base_position == (0.0, 0.0, 0.0)


def test_get_base_velocity():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    pybullet.step()
    base_velocity = pybullet.get_base_velocity("my_box")
    pybullet.close()
    assert pytest.approx(list(base_velocity), abs=0.001) == [0.0, 0.0, -0.392]


def test_get_base_orientation():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    base_orientation = pybullet.get_base_orientation("my_box")
    pybullet.close()
    assert pytest.approx(list(base_orientation), abs=0.001) == [0.0, 0.0, 0.0, 1.0]


def test_get_base_rotation():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    base_rotation = pybullet.get_base_rotation("my_box")
    print(base_rotation)
    pybullet.close()
    assert pytest.approx(list(base_rotation), abs=0.001) == [0.0, 0.0, 0.0]


def test_get_base_angular_velocity():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    base_angular_velocity = pybullet.get_base_angular_velocity("my_box")
    pybullet.close()
    assert pytest.approx(list(base_angular_velocity), abs=0.001) == [0.0, 0.0, 0.0]


def test_load_URDF():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.close()


def test_control_joints():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.control_joints("panda", [5], [0.3], [5.0])
    pybullet.step()


def test_get_link_position():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    link_position = pybullet.get_link_position("panda", 1)
    assert pytest.approx(list(link_position), abs=0.001) == [0.000, 0.060, 0.373]
    pybullet.close()


def test_get_link_orientation():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.control_joints("panda", [5], [0.3], [5.0])
    pybullet.step()
    link_orientation = pybullet.get_link_orientation("panda", 5)
    assert pytest.approx(list(link_orientation), abs=0.001) == [0.707, -0.02, 0.02, 0.707]
    pybullet.close()


def test_get_link_velocity():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.control_joints("panda", [5], [0.3], [5.0])
    pybullet.step()
    link_velocity = pybullet.get_link_velocity("panda", 5)
    assert pytest.approx(list(link_velocity), abs=0.001) == [-0.0068, 0.0000, 0.1186]
    pybullet.close()


def test_get_link_angular_velocity():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.control_joints("panda", [5], [0.3], [5.0])
    pybullet.step()
    link_angular_velocity = pybullet.get_link_angular_velocity("panda", 5)
    assert pytest.approx(list(link_angular_velocity), abs=0.001) == [0.000, -2.969, 0.000]
    pybullet.close()


def test_get_joint_angle():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.control_joints("panda", [5], [0.3], [5.0])
    pybullet.step()
    joint_angle = pybullet.get_joint_angle("panda", 5)
    assert pytest.approx(joint_angle, abs=0.001) == 0.063
    pybullet.close()


def test_set_base_pose():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    pybullet.set_base_pose("my_box", [1.0, 1.0, 1.0], [0.707, -0.02, 0.02, 0.707])
    base_position = pybullet.get_base_position("my_box")
    base_orientation = pybullet.get_base_orientation("my_box")
    pybullet.close()
    assert pytest.approx(list(base_position), abs=0.001) == [1.0, 1.0, 1.0]
    assert pytest.approx(list(base_orientation), abs=0.001) == [0.707, -0.02, 0.02, 0.707]


def test_set_joint_angle():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.loadURDF(
        body_name="panda",
        fileName="franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    pybullet.set_joint_angle("panda", 3, 0.4)
    joint_angle = pybullet.get_joint_angle("panda", 3)
    assert pytest.approx(joint_angle, abs=0.001) == 0.4
    pybullet.close()
