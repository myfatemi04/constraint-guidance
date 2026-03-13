from dataclasses import dataclass

import numpy as np

from ael.problem import Problem


@dataclass
class ConstraintSatisfaction:
    agent_circular_obstacle_constraint_residuals: np.ndarray
    """ Residuals for agent-obstacle nonpenetration constraints. If positive, represents the amount by which the signed distance into the obstacle exceeds zero. Shape ([b], t, num_agents, num_obstacles). """

    agent_rectangular_obstacle_constraint_residuals: np.ndarray
    """ Residuals for agent-axis-aligned-box-obstacle nonpenetration constraints. If positive, represents the amount by which the signed distance into the obstacle exceeds zero. Shape ([b], t, num_agents, num_axis_aligned_box_obstacles). """

    agent_agent_constraint_residuals: np.ndarray
    """ Residuals for agent-agent nonpenetration constraints. If positive, represents the amount by which the signed distance into another agent exceeds zero. Shape ([b], t, num_agents, num_agents). """

    velocity_constraint_residuals: np.ndarray
    """ Residuals for velocity constraints. If positive, represents the amount by which the velocity exceeds the maximum allowed velocity. Shape ([b], t-1, num_agents). """


def compute_agent_circular_obstacle_constraint_residuals(
    problem: Problem[np.ndarray], trajectory_b_T_A_D: np.ndarray
) -> np.ndarray:
    if problem.num_circular_obstacles == 0:
        return np.zeros(
            trajectory_b_T_A_D.shape[:-1] + (problem.num_circular_obstacles,)
        )

    ### Agent-obstacle nonpenetration constraints ###
    agent_obstacle_distances_b_T_A_O = (
        # ([b], t, num_agents, 1, 2) - (1, num_obstacles, 2) -> ([b], t, num_agents, num_obstacles, 2)
        trajectory_b_T_A_D[..., :, np.newaxis, :]
        - problem.circular_obstacle_positions[np.newaxis, :, :]
    )
    agent_obstacle_distances_b_T_A_O = np.linalg.norm(
        agent_obstacle_distances_b_T_A_O, axis=-1
    )
    min_acceptable_agent_obstacle_distances_A_O = (
        # (num_agents, 1) + (1, num_obstacles) -> (num_agents, num_obstacles)
        problem.agent_radii[:, np.newaxis]
        + problem.circular_obstacle_radii[np.newaxis, :]
    )
    agent_obstacle_constraint_residuals_b_T_A_O = (
        min_acceptable_agent_obstacle_distances_A_O - agent_obstacle_distances_b_T_A_O
    )
    agent_obstacle_constraint_residuals_b_T_A_O[
        agent_obstacle_constraint_residuals_b_T_A_O < 0
    ] = 0.0

    return agent_obstacle_constraint_residuals_b_T_A_O


def compute_agent_rectangular_obstacle_constraint_residuals(
    problem: Problem[np.ndarray], trajectory_b_T_A_D: np.ndarray
) -> np.ndarray:
    # 2 = (low, high)
    bounds_O_2_D = problem.axis_aligned_box_obstacle_bounds
    distances_b_T_A_O_2_D = trajectory_b_T_A_D[..., :, :, None, None, :] - bounds_O_2_D
    # distance from lower bound is negated
    distances_b_T_A_O_2_D[..., 0, :] *= -1
    distances_b_T_A_O = np.max(np.max(distances_b_T_A_O_2_D, axis=-1), axis=-1)
    distances_b_T_A_O -= problem.agent_radii[:, np.newaxis]
    distances_b_T_A_O[distances_b_T_A_O > 0] = 0
    return -distances_b_T_A_O


def compute_agent_agent_constraint_residuals(
    problem: Problem[np.ndarray], trajectory_b_T_A_D: np.ndarray
) -> np.ndarray:
    ### Agent-agent nonpenetration constraints ###
    agent_distances_b_T_A_A = (
        # ([b], t, num_agents, 1, 2) - ([b], t, 1, num_agents, 2) -> ([b], t, num_agents, num_agents, 2)
        trajectory_b_T_A_D[..., :, np.newaxis, :]
        - trajectory_b_T_A_D[..., np.newaxis, :, :]
    )
    # ([b], t, num_agents, num_agents)
    agent_distances_b_T_A_A = np.linalg.norm(agent_distances_b_T_A_A, axis=-1)
    min_acceptable_distances_A_A = (
        problem.agent_radii[:, np.newaxis] + problem.agent_radii[np.newaxis, :]
    )
    agent_agent_constraint_residuals_b_T_A_A = (
        min_acceptable_distances_A_A - agent_distances_b_T_A_A
    )
    agent_agent_constraint_residuals_b_T_A_A[
        agent_agent_constraint_residuals_b_T_A_A < 0
    ] = 0.0
    diag_indices = np.diag_indices(agent_agent_constraint_residuals_b_T_A_A.shape[-2])
    agent_agent_constraint_residuals_b_T_A_A[..., diag_indices[0], diag_indices[1]] = (
        0.0
    )
    assert (
        np.diagonal(agent_agent_constraint_residuals_b_T_A_A, axis1=-2, axis2=-1) == 0
    ).all()

    return agent_agent_constraint_residuals_b_T_A_A


def compute_velocity_constraint_residuals(
    problem: Problem[np.ndarray], trajectory_b_T_A_D: np.ndarray
) -> np.ndarray:
    ### Velocity constraints ###
    # ([b], t, num_agents, 2)
    velocity_b_T_A_D = (
        trajectory_b_T_A_D[..., 1:, :, :] - trajectory_b_T_A_D[..., :-1, :, :]
    )
    speed_b_T_A_D = np.linalg.norm(velocity_b_T_A_D, axis=-1)
    velocity_constraint_residual_b_T_A_D = speed_b_T_A_D - problem.agent_max_speeds
    velocity_constraint_residual_b_T_A_D[velocity_constraint_residual_b_T_A_D < 0] = 0.0

    return velocity_constraint_residual_b_T_A_D


def compute_constraint_residuals(
    problem: Problem[np.ndarray], trajectory_b_T_A_D: np.ndarray
) -> ConstraintSatisfaction:
    """
    Return the constraint residuals for agent-obstacle and agent-agent distances, respectively.
    Allows computation for single trajectories and/or batches of trajectories. A residual > 0
    indicates a constraint violation.
    """
    return ConstraintSatisfaction(
        agent_circular_obstacle_constraint_residuals=compute_agent_circular_obstacle_constraint_residuals(
            problem, trajectory_b_T_A_D
        ),
        agent_rectangular_obstacle_constraint_residuals=compute_agent_rectangular_obstacle_constraint_residuals(
            problem, trajectory_b_T_A_D
        ),
        agent_agent_constraint_residuals=compute_agent_agent_constraint_residuals(
            problem, trajectory_b_T_A_D
        ),
        velocity_constraint_residuals=compute_velocity_constraint_residuals(
            problem, trajectory_b_T_A_D
        ),
    )
