import numpy as np

from apairo_huggingface.poses import executed_actions, relative_states


def _arc(n=12, speed=2.0, yaw_rate=0.3, dt=0.1):
    # A robot on a circle, starting away from the world origin, heading 40 deg.
    poses, yaw, xy = [], np.radians(40.0), np.array([5.0, -3.0])
    for _ in range(n):
        T = np.eye(4)
        T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        T[:2, 3] = xy
        T[2, 3] = 1.5
        poses.append(T)
        xy = xy + speed * dt * np.array([np.cos(yaw), np.sin(yaw)])
        yaw += yaw_rate * dt
    return np.stack(poses), np.arange(n) * dt


def test_states_start_at_the_identity_of_their_first_pose():
    poses, _ = _arc()
    states = relative_states(poses)
    assert states.dtype == np.float32 and states.shape == (12, 7)
    np.testing.assert_allclose(states[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(states[1, :3], [0.2, 0, 0], atol=1e-6)        # 2 m/s * 0.1 s straight ahead
    # An episode cut from the middle starts at the identity too.
    np.testing.assert_allclose(relative_states(poses[5:])[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(relative_states(poses[5:])[1], states[1], atol=1e-5)


def test_actions_are_the_executed_speed_and_yaw_rate():
    poses, stamps = _arc()
    actions = executed_actions(poses, stamps)
    assert actions.dtype == np.float32 and actions.shape == (12, 2)
    np.testing.assert_allclose(actions, np.tile([2.0, 0.3], (12, 1)), atol=1e-5)
    # Across +-pi the yaw is unwrapped.
    poses, stamps = _arc(yaw_rate=3.0)
    np.testing.assert_allclose(executed_actions(poses, stamps)[:, 1], 3.0, atol=1e-4)
