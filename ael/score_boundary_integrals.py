import numpy as np

from ael.problem import Problem

ARC_CENTER_X_DIM = 0
ARC_CENTER_Y_DIM = 1
ARC_RADIUS_DIM = 2
ARC_THETA1_DIM = 3
ARC_THETA2_DIM = 4
ARC_SIGN_DIM = 5

DISK_CENTER_X_DIM = 0
DISK_CENTER_Y_DIM = 1
DISK_RADIUS_DIM = 2
DISK_SIGN_DIM = 3


def compute_feasibility_score_numerator(
    xy: np.ndarray, sigma: float, boundaries: np.ndarray, n_points: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    An improved approach for computing the score using line integrals.

    Args:
    - xy: (N, 2) array of points at which to evaluate the score.
    - sigma: stddev for Gaussian.
    - surfaces: (N, S, 6) array of (center_x, center_y, radius, theta1, theta2, sign). The first dimension may be 1 if the surfaces are shared across all inputs.

    The sign indicates whether it is an "exclusion" or "inclusion" constraint. These constraint
    surfaces can be created by computing the intersection points between each of the obstacles.
    Here, it's assumed that the constraint surfaces are piecewise arcs, because the feasible set
    is created through the intersection of disks, which have boundaries which are arcs. The
    denominator is estimated by sampling several points around the current location and checking
    for feasibility. For now, if it's found to be too low, the numerator will be normalized to
    have a magnitude of 1.
    """

    # (N, S). S is the number of surfaces.
    center_x = boundaries[:, :, ARC_CENTER_X_DIM]
    center_y = boundaries[:, :, ARC_CENTER_Y_DIM]
    radius = boundaries[:, :, ARC_RADIUS_DIM]
    theta1 = boundaries[:, :, ARC_THETA1_DIM]
    theta2 = boundaries[:, :, ARC_THETA2_DIM]
    sign = boundaries[:, :, ARC_SIGN_DIM]

    # (N, T, S). T is the number of points sampled along each surface.
    theta = np.linspace(theta1, theta2, num=n_points, endpoint=False, axis=1)
    n_x = np.cos(theta)
    n_y = np.sin(theta)
    # (N, S)
    dtheta = theta[:, 1, :] - theta[:, 0, :]
    # (N, T, S) <- (N, :, S) + (N, :, S) * (N, T, S)
    x = center_x[:, np.newaxis, :] + radius[:, np.newaxis, :] * n_x
    y = center_y[:, np.newaxis, :] + radius[:, np.newaxis, :] * n_y
    # (N, S) <- (N, S) * (N, S)
    ds = dtheta * radius

    # (N, T, S) <- (N, T, S) - (N, :, :). N is the number of points at which the score is being evaluated.
    delta_x = x - xy[:, 0, np.newaxis, np.newaxis]
    delta_y = y - xy[:, 1, np.newaxis, np.newaxis]
    exp_arg = -0.5 * (delta_x**2 + delta_y**2) / (sigma**2)
    log_scaler = exp_arg.max(axis=-1).max(axis=-1)
    exp_arg_adjusted = exp_arg - log_scaler[:, np.newaxis, np.newaxis]

    gaussian_pdf = np.exp(exp_arg_adjusted) / (2 * np.pi * sigma**2)

    # (N, T, S) <- (N, :, S) * (N, T, S) * (N, :, S) * (N, T, S)
    integrand_x = sign[:, np.newaxis, :] * gaussian_pdf * ds[:, np.newaxis, :] * n_x
    integrand_y = sign[:, np.newaxis, :] * gaussian_pdf * ds[:, np.newaxis, :] * n_y

    # (N,) <- sum over T and S of (N, T, S)
    integral_x = np.sum(np.sum(integrand_x, axis=-1), axis=-1)
    integral_y = np.sum(np.sum(integrand_y, axis=-1), axis=-1)

    """
    delta_min can be used to compare with the denominator.
    If the denominator is large, the numerator can essentially be discarded.
    If the denominator is small, the delta_min can be discarded and the numerator
    can be normalized to have a magnitude of 1, since the direction is still
    informative.
    """

    return np.stack([integral_x, integral_y], axis=-1), log_scaler


def compute_feasibility_score_denominator(
    xy: np.ndarray, sigma: float, disks: np.ndarray, n_samples: int = 100
):
    """
    Computes feasibility by sampling around the current point and checking whether the results
    lie inside the disks or not. The disks are (center_x, center_y, radius, sign) tuples. There
    is no theta 1 or theta 2 in this case.

    xy: (N, 2) - N is the number of points for which the score is being evaluated.
    """

    # (S,). S is the number of disks.
    center_x = disks[:, 0]
    center_y = disks[:, 1]
    radius = disks[:, 2]
    sign = disks[:, 3]

    # (B, N, 2). B is the number of samples, N is the number of points, 2 is for x and y.
    xy_noise = (
        xy[np.newaxis, :, :] + np.random.normal(size=(n_samples, *xy.shape)) * sigma
    )
    # (B, N, S) <- (B, N, :) - (S)
    xy_distances = np.sqrt(
        (xy_noise[:, :, np.newaxis, 0] - center_x) ** 2
        + (xy_noise[:, :, np.newaxis, 1] - center_y) ** 2
    )

    # (B, N, S) <- (B, N, S) compared to (S)
    inside = xy_distances <= radius
    outside = xy_distances >= radius

    # (B, N) <- all(axis=-1) of (B, N, S) <- (B, N, S) & (S)
    ok = np.all(outside & (sign == 1) | inside & (sign == -1), axis=-1)
    # (N,) <- mean(axis=0) of (B, N)
    ok_fraction = ok.mean(axis=0)

    return ok_fraction


def compute_score_from_boundary_integrals(
    trajectory: np.ndarray,
    problem: Problem[np.ndarray],
    sigma: float,
    obstacle_boundaries: np.ndarray,
    include_kinetic: bool,
):
    score = np.zeros_like(trajectory)
    for agent_i in range(trajectory.shape[1]):
        num_other_agents = trajectory.shape[1] - 1

        # Initialize surfaces with obstacles.
        boundaries = np.zeros(
            (trajectory.shape[0], obstacle_boundaries.shape[0] + num_other_agents, 6)
        )
        boundaries[:, : obstacle_boundaries.shape[0], :] = obstacle_boundaries
        boundaries[:, : obstacle_boundaries.shape[0], ARC_RADIUS_DIM] += (
            problem.agent_radii[agent_i]
        )

        # Initialize disks with obstacles. We must select indices carefully here because we're translating between two array formats.
        disks = np.zeros(
            (trajectory.shape[0], obstacle_boundaries.shape[0] + num_other_agents, 4)
        )
        disks[
            :,
            : obstacle_boundaries.shape[0],
            [DISK_CENTER_X_DIM, DISK_CENTER_Y_DIM, DISK_RADIUS_DIM, DISK_SIGN_DIM],
        ] = obstacle_boundaries[
            :, [ARC_CENTER_X_DIM, ARC_CENTER_Y_DIM, ARC_RADIUS_DIM, ARC_SIGN_DIM]
        ]
        disks[:, : obstacle_boundaries.shape[0], DISK_RADIUS_DIM] += (
            problem.agent_radii[agent_i]
        )

        # Include other agents as surfaces and disks.
        target_indices = []
        source_indices = []
        index = obstacle_boundaries.shape[0]
        for other_agent_i in range(trajectory.shape[1]):
            if other_agent_i == agent_i:
                continue

            source_indices.append(other_agent_i)
            target_indices.append(index)
            index += 1

        # Create new surfaces for agent-agent interactions.
        # These are circles around the other agents with radius equal to the sum of the agent radii.
        other_agent_trajectories = trajectory[:, source_indices, :]
        r = problem.agent_radii[agent_i] + problem.agent_radii[source_indices]
        boundaries[:, target_indices, ARC_CENTER_X_DIM] = other_agent_trajectories[
            :, :, 0
        ]
        boundaries[:, target_indices, ARC_CENTER_Y_DIM] = other_agent_trajectories[
            :, :, 1
        ]
        boundaries[:, target_indices, ARC_RADIUS_DIM] = r
        boundaries[:, target_indices, ARC_THETA1_DIM] = 0
        boundaries[:, target_indices, ARC_THETA2_DIM] = 2 * np.pi
        boundaries[:, target_indices, ARC_SIGN_DIM] = 1.0

        disks[:, target_indices, DISK_CENTER_X_DIM] = other_agent_trajectories[:, :, 0]
        disks[:, target_indices, DISK_CENTER_Y_DIM] = other_agent_trajectories[:, :, 1]
        disks[:, target_indices, DISK_RADIUS_DIM] = r
        disks[:, target_indices, DISK_SIGN_DIM] = 1.0

        xy = trajectory[:, agent_i, :]

        eps = 1e-12
        numer, log_numer_scaler = compute_feasibility_score_numerator(
            xy, sigma, boundaries
        )
        denom = compute_feasibility_score_denominator(xy, sigma, disks)

        numer_scaler = np.maximum(np.exp(log_numer_scaler), eps * (denom < 0.5))
        numer_scaler = np.exp(log_numer_scaler)

        obs_feas_score = (
            numer * numer_scaler[:, np.newaxis] / (denom[:, np.newaxis] + eps)
        )
        obs_feas_score_mag = np.linalg.norm(obs_feas_score, axis=-1, keepdims=True)
        obs_feas_score = obs_feas_score / np.maximum(obs_feas_score_mag, 1.0)

        # Object feasibility component.
        score[:, agent_i, :] = obs_feas_score

        # Kinetic energy component.
        if include_kinetic:
            mu = (trajectory[2:, agent_i, :] + trajectory[:-2, agent_i, :]) / 2
            score[1:-1, agent_i, :] += (
                1 / (1 + sigma**2) * (mu - trajectory[1:-1, agent_i, :])
            ) * 0.5

    return score
