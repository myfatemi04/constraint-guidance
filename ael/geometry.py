import numpy as np

from ael.problem import Problem


def circle_circle_intersection_angles(A, B):
    """
    Compute intersection-point angles for all pairs between two circle lists.

    Parameters:
    - A: ndarray of n * (x, y, r) for set A
    - B: ndarray of m * (x, y, r) for set B

    Returns:
    - angles: ndarray of (N, M, 4). Intersection angles on circles in A and B, or NaN if no intersection
    """

    # (N,). N is the number of circles in A.
    ax, ay, ar = A[:, 0], A[:, 1], A[:, 2]
    # (M,). M is the number of circles in B.
    bx, by, br = B[:, 0], B[:, 1], B[:, 2]

    # (N, M)
    dx = bx[np.newaxis, :] - ax[:, np.newaxis]
    dy = by[np.newaxis, :] - ay[:, np.newaxis]
    d = np.sqrt(dx**2 + dy**2)
    rsum = ar[:, np.newaxis] + br[np.newaxis, :]
    rdiff = np.abs(ar[:, np.newaxis] - br[np.newaxis, :])
    valid = (d <= rsum) & (d >= rdiff)

    # Standard circle-circle intersection math
    a = (ar[:, np.newaxis] ** 2 - br[np.newaxis, :] ** 2 + d**2) / (2.0 * d)
    h2 = ar[:, np.newaxis] ** 2 - a**2

    # Clamp h2 to 0 for near-tangent numerical negatives
    h2[h2 < 0.0] = 0.0
    h = np.sqrt(h2)

    # Point along the center-center line (base point)
    x2 = ax[:, np.newaxis] + a * dx / d
    y2 = ay[:, np.newaxis] + a * dy / d

    # Offset perpendicular to center-center line
    rx = -dy / d
    ry = dx / d

    # Two intersection points for each pair (N, M)
    x3p = x2 + h * rx
    y3p = y2 + h * ry
    x3m = x2 - h * rx
    y3m = y2 - h * ry

    # Angles on circle A
    ang_a_p = np.arctan2(y3p - ay[:, np.newaxis], x3p - ax[:, np.newaxis])
    ang_a_m = np.arctan2(y3m - ay[:, np.newaxis], x3m - ax[:, np.newaxis])

    # Angles on circle B
    ang_b_p = np.arctan2(y3p - by[np.newaxis, :], x3p - bx[np.newaxis, :])
    ang_b_m = np.arctan2(y3m - by[np.newaxis, :], x3m - bx[np.newaxis, :])

    # Pack and mask invalids with NaN
    angles_a = np.stack([ang_a_p, ang_a_m], axis=-1)
    angles_b = np.stack([ang_b_p, ang_b_m], axis=-1)

    angles_a = np.where(valid[..., np.newaxis], angles_a, np.nan)
    angles_b = np.where(valid[..., np.newaxis], angles_b, np.nan)

    return np.concatenate([angles_a, angles_b], axis=-1)


def compute_obstacle_boundaries(problem: Problem[np.ndarray]) -> np.ndarray:
    """
    Returns a list of surfaces corresponding to the arcs of each obstacle circle that are not occluded by other obstacles. Each surface is represented as a tuple of (center_x, center_y, radius, theta_start, theta_end, sign), where sign is +1 for exclusion constraints and -1 for inclusion constraints.
    """

    obstacle_circles = np.concatenate(
        [
            problem.obstacle_positions,
            problem.obstacle_radii[:, np.newaxis],
        ],
        axis=1,
    )
    obstacle_intersection_angles = circle_circle_intersection_angles(
        obstacle_circles, obstacle_circles
    )

    # Identify the distinct angles for each obstacle. By symmetry, switching obstacles just switches the two
    # pairs of angles, so we can just look at the first two columns corresponding to the first set of circles.
    angles = obstacle_intersection_angles[:, :, :2].reshape(
        obstacle_intersection_angles.shape[0], -1
    )
    # (o, s)
    angles = [np.sort(a[~np.isnan(a)]) for a in angles]
    # (o, s, 2) corresponding to start and end thetas for each surface.
    spans = [
        np.stack([a, np.array([*a[1:], a[0] + 2 * np.pi])], axis=-1) for a in angles
    ]
    # Compute midpoints of each span. These are used to identify the feasibility of each surface by checking the midpoint against other obstacles.
    midpoint_thetas = [(s[:, 0] + s[:, 1]) / 2 for s in spans]
    midpoints = [
        np.stack(
            [
                # x
                obstacle_circles[i, 0]
                + obstacle_circles[i, 2] * np.cos(midpoint_thetas[i]),
                # y
                obstacle_circles[i, 1]
                + obstacle_circles[i, 2] * np.sin(midpoint_thetas[i]),
            ],
            axis=-1,
        )
        for i in range(len(problem.obstacle_positions))
    ]
    # (o, s, o)
    midpoint_signed_distances = [
        # (s, o) <- (s, o) - (:, o)
        np.linalg.norm(
            # (s, o, 2) <- (s, :, 2) - (:, o, 2)
            midpoints[i][:, np.newaxis, :] - obstacle_circles[np.newaxis, :, :2],
            axis=-1,
        )
        - obstacle_circles[np.newaxis, :, 2]
        for i in range(len(problem.obstacle_positions))
    ]
    for i in range(len(problem.obstacle_positions)):
        # We check if we should delete an arc by checking if the signed distance between the midpoint and the other obstacle is negative.
        # However, we need to ignore the distance to the obstacle itself, just to avoid misclassifications in case of numerical issues.
        midpoint_signed_distances[i][:, i] = 1.0

    # (o, s) boolean of whether each surface is valid (not inside another obstacle)
    midpoint_inclusions = [
        np.any(midpoint_signed_distances[i] < 0, axis=-1)
        for i in range(len(problem.obstacle_positions))
    ]

    # (o, s, 2) of start and end angles for each valid surface.
    obstacle_surfaces = [
        spans[i][midpoint_inclusions[i]] for i in range(len(problem.obstacle_positions))
    ]

    n_surfaces = sum(len(s) for s in obstacle_surfaces)
    result = np.zeros((n_surfaces, 6))
    result[:, 5] = (
        1.0  # default sign is positive, because all object circles are exclusion constraints.
    )

    pointer = 0
    for i in range(len(problem.obstacle_positions)):
        # (center_x, center_y, radius, theta1, theta2, sign)
        result[pointer : pointer + len(obstacle_surfaces[i]), :3] = obstacle_circles[i]
        result[pointer : pointer + len(obstacle_surfaces[i]), 3:5] = obstacle_surfaces[
            i
        ]

        pointer += len(obstacle_surfaces[i])

    return result
