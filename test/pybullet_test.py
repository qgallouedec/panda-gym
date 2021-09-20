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
    assert base_position == (0.0, 0.0, 0.0)
    pybullet.close()


def test_get_base_velocity():
    from panda_gym.pybullet import PyBullet

    pybullet = PyBullet()
    pybullet.create_box("my_box", [0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0])
    pybullet.step()
    base_velocity = pybullet.get_base_velocity("my_box")
    assert pytest.approx(list(base_velocity), 0.001) == [0.0, 0.0, -0.392]
    pybullet.close()
