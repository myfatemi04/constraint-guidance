import numpy as np

from ael.problem import Problem


def _compute_bounds(centers: np.ndarray, sizes: np.ndarray):
    return np.stack([centers - sizes / 2, centers + sizes / 2], axis=1)


BUILT_IN_MAPS = {
    # https://github.com/yoraish/mmd/blob/main/deps/torch_robotics/torch_robotics/environments/env_conveyor_2d.py
    "conveyor_2d": {
        "circular_obstacle_positions": [],
        "circular_obstacle_radii": [],
        "axis_aligned_obstacle_bounds": _compute_bounds(
            np.array([[0, 0], [0, 0.35], [0, -0.35]]),
            np.array([[0.8, 0.1], [1.0, 0.1], [1.0, 0.1]]),
        ),
    },
    # https://github.com/yoraish/mmd/blob/main/deps/torch_robotics/torch_robotics/environments/env_empty_2d.py
    "empty_2d": {
        "circular_obstacle_positions": [],
        "circular_obstacle_radii": [],
        "axis_aligned_obstacle_bounds": [],
    },
    # https://github.com/yoraish/mmd/blob/main/deps/torch_robotics/torch_robotics/environments/env_highways_2d.py
    "highways": {
        "circular_obstacle_positions": [],
        "circular_obstacle_radii": [],
        "axis_aligned_obstacle_bounds": _compute_bounds(
            np.array(
                [
                    [0, 0.0],
                    [0.0, 0.875],
                    [0.0, -0.875],
                    [0.875, 0.0],
                    [-0.875, 0.0],
                    [0.875, 0.875],
                    [0.875, -0.875],
                    [-0.875, 0.875],
                    [-0.875, -0.875],
                ]
            ),
            np.array(
                [
                    [0.5, 0.5],
                    [0.5, 0.25],
                    [0.5, 0.25],
                    [0.25, 0.5],
                    [0.25, 0.5],
                    [0.25, 0.25],
                    [0.25, 0.25],
                    [0.25, 0.25],
                    [0.25, 0.25],
                ]
            ),
        ),
    },
}


# These general methods are used in the original codebase in https://github.com/yoraish/mmd/blob/main/mmd/config/mmd_experiment_configs.py.


# Adapted from https://github.com/yoraish/mmd/blob/main/mmd/common/multi_agent_utils.py
def get_start_goal_pos_boundary(num_agents: int, dist=0.87):
    start = np.array(
        [
            [
                dist * np.cos(2 * np.pi * i / num_agents),
                dist * np.sin(2 * np.pi * i / num_agents),
            ]
            for i in range(num_agents)
        ]
    ).clip(-1, 1)

    goal = start.copy()
    # flip X if abs(X) < abs(Y) else flip Y
    flip_x_mask = np.abs(start[:, 0]) < np.abs(start[:, 1])
    goal[flip_x_mask, 0] *= -1
    goal[~flip_x_mask, 1] *= -1
    return start, goal


def is_collision_free(
    problem: Problem, point: np.ndarray, min_point_obstacle_distance: float
) -> bool:
    circular_positions = problem.circular_obstacle_positions
    circular_radii = problem.circular_obstacle_radii
    axis_bounds = problem.axis_aligned_box_obstacle_bounds

    if circular_positions.size > 0:
        distances = np.linalg.norm(point - circular_positions, axis=1)
        if np.any(distances <= circular_radii + min_point_obstacle_distance):
            return False

    if axis_bounds.size > 0:
        lower = axis_bounds[:, 0, :] - min_point_obstacle_distance
        upper = axis_bounds[:, 1, :] + min_point_obstacle_distance
        inside_box = np.all((point >= lower) & (point <= upper), axis=1)
        if np.any(inside_box):
            return False

    return True


# Adapted from https://github.com/yoraish/mmd/blob/main/mmd/common/multi_agent_utils.py
def generate_positions_random(
    problem: Problem,
    tensor_args,
    min_point_point_distance=0.15,
    min_point_obstacle_distance=0.16,
    box_size=0.95,
):
    def sample_set() -> list[np.ndarray]:
        points = []
        for _ in range(problem.num_agents):
            while True:
                point = (np.random.rand(2).astype(np.float32) * 2 - 1) * box_size
                if (
                    is_collision_free(problem, point, min_point_obstacle_distance)
                    and len(points) == 0
                    or np.any(
                        np.linalg.norm(point - np.asarray(points), axis=1)
                        <= min_point_point_distance
                    )
                ):
                    points.append(point)
                    break
        return points

    return sample_set(), sample_set()
