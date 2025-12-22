from typing import Literal
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def squared_distance_to_line_segment(x1, x2, y):
    s = torch.clamp(
        ((y - x1) * (x2 - x1)).sum(dim=-1) / ((x2 - x1) ** 2).sum(dim=-1), 0, 1
    ).detach()
    xs = x1 + s * (x2 - x1)
    return (xs - y).pow(2).sum(dim=-1)


class Path:
    def __init__(self, vertices: torch.Tensor):
        # list of (x, y, delta_t)
        self.vertices = vertices

    def compute_velocity_constraint(self, max_velocity: float):
        dx_square = self.vertices[1:, :2] - self.vertices[:-1, :2]
        dt = self.vertices[:-1, 2]
        speed_squared = (dx_square**2).sum(dim=1) / dt**2
        return torch.maximum(
            (speed_squared - max_velocity**2) * dt, torch.zeros_like(speed_squared)
        )

    def compute_min_time_objective(self):
        return self.vertices[:-1, 2].sum()  # ignore the last vertex, it's the end time

    def compute_simplicity_objective(self):
        return len(self.vertices) - 2

    def can_remove(self, i: int, theta_thres, d_thres):
        prev_vector = self.vertices[i] - self.vertices[i - 1]
        next_vector = self.vertices[i + 1] - self.vertices[i]
        dot = (prev_vector * next_vector).sum(dim=-1)
        norm = torch.norm(prev_vector) * torch.norm(next_vector)
        cos_theta = dot / norm
        distance = squared_distance_to_line_segment(
            self.vertices[i - 1][:2], self.vertices[i + 1][:2], self.vertices[i][:2]
        )
        return (
            cos_theta < torch.cos(torch.tensor(theta_thres)) and distance < d_thres**2
        )

    def remove(self, i: int):
        # assumes can_remove check already passed
        before = self.vertices[:i].clone()
        before[i - 1][2] += self.vertices[i][2].detach()
        after = self.vertices[i + 1 :]
        new_vertices = torch.cat([before, after], dim=0)
        return Path(new_vertices)

    def insert(self, i: int):
        before = self.vertices[:i]
        after = self.vertices[i + 1 :]
        delta_t = self.vertices[i][2] / 2
        new_vertices = torch.cat(
            [
                before,
                torch.tensor([[self.vertices[i][0], self.vertices[i][1], delta_t]]),
                torch.tensor(
                    [
                        [
                            (self.vertices[i][0] + self.vertices[i + 1][0]) / 2,
                            (self.vertices[i][1] + self.vertices[i + 1][1]) / 2,
                            delta_t,
                        ]
                    ]
                ),
                after,
            ],
            dim=0,
        )
        return Path(new_vertices.detach())

    def propose_rewrites(self, k=64, theta_thres=torch.pi / 12, d_thres=0.1):
        rewrites = []
        for i in range(1, len(self.vertices) - 1):
            if self.can_remove(i, theta_thres, d_thres):
                rewrites.append(("remove", i))

        for i in range(len(self.vertices) - 1):
            rewrites.append(("insert", i))

        # Uniformly sample k rewrites
        return [rewrites[i] for i in torch.randperm(len(rewrites))[:k]]

    def apply_rewrite(self, rewrite: tuple[str, int]):
        if rewrite[0] == "remove":
            return self.remove(rewrite[1])
        elif rewrite[0] == "insert":
            return self.insert(rewrite[1])
        else:
            raise ValueError(f"Invalid rewrite: {rewrite}")

    def apply_gradient(self, grad: torch.Tensor, learning_rate: float):
        middle_vertices = self.vertices[1:-1] - (learning_rate * grad[1:-1])
        return Path(
            torch.cat(
                [self.vertices[:1], middle_vertices, self.vertices[-1:]], dim=0
            ).detach()
        )


class Map:
    def __init__(self, obstacles: torch.Tensor):
        # (x, y, radius)
        self.obstacles = obstacles

    def compute_collision_constraint(self, path: Path, agent_radius: float):
        constraints = []
        for i in range(len(path.vertices) - 1):
            start = path.vertices[i][:2]
            end = path.vertices[i + 1][:2]
            for obstacle in self.obstacles:
                obstacle_center = obstacle[:2]
                obstacle_radius = obstacle[2]
                squared_distance = squared_distance_to_line_segment(
                    start, end, obstacle_center
                )
                constraint = torch.maximum(
                    (obstacle_radius + agent_radius) ** 2 - squared_distance,
                    torch.zeros_like(squared_distance),
                )
                constraints.append(constraint)
        return torch.stack(constraints).view(
            len(path.vertices) - 1, len(self.obstacles)
        )


def render(
    map: Map,
    paths: list[Path],
    colors: list[str],
    path_grads: list[torch.Tensor | None] | None = None,
    pause_behavior: Literal["show", "pause"] = "show",
):
    plt.figure()

    # Draw circles with correct radius for obstacles.
    for obstacle in map.obstacles:
        plt.gca().add_patch(
            patches.Circle(
                (obstacle[0].item(), obstacle[1].item()),
                obstacle[2].item(),
                color="red",
                alpha=0.5,
            )
        )

    for i, path in enumerate(paths):
        vertices = path.vertices.detach()

        grad: torch.Tensor = (
            path_grads[i]  # type: ignore
            if path_grads is not None and path_grads[i] is not None
            else None
        )

        # Draw path.
        plt.plot(vertices[:, 0], vertices[:, 1], color=colors[i])

        # Draw circles for path keypoints.
        for t, vertex in enumerate(vertices):
            plt.scatter(vertex[0], vertex[1], color=colors[i], s=10)

            if grad is not None:
                plt.quiver(
                    vertex[0],
                    vertex[1],
                    grad[t, 0],
                    grad[t, 1],
                    color=colors[i],
                    scale=10,
                )

    plt.axis("equal")
    if pause_behavior == "show":
        plt.show()
    elif pause_behavior == "pause":
        plt.pause(0.02)
    plt.close()


def compute_objectives(
    path: Path, map: Map, agent_radius: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity_constraint = path.compute_velocity_constraint(agent_radius)
    collision_constraint = map.compute_collision_constraint(path, agent_radius)
    objective = path.compute_min_time_objective()
    return velocity_constraint, collision_constraint, objective


def compute_grad(
    path: Path, map: Map, agent_radius: float, constraint_rho: float
) -> torch.Tensor:
    path.vertices.requires_grad_()

    velocity_constraint, collision_constraint, objective = compute_objectives(
        path, map, agent_radius
    )

    loss = velocity_constraint.sum() + collision_constraint.sum()
    (loss * constraint_rho + objective).backward()

    grad = path.vertices.grad  # type: ignore

    assert grad is not None

    return grad


def main():
    # Start with a simple path, going from (0, 0) to (0, 10) in 5 seconds. Maximum velocity is 2 m/s.
    path = Path(torch.tensor([[0, 0, 5], [0, 10, 0]], dtype=torch.float32))
    map = Map(
        torch.tensor(
            [
                [0.2, 5.0, 0.5],
                [-0.2, 8.0, 0.5],
            ],
            dtype=torch.float32,
        )
    )
    agent_radius = 0.2
    learning_rate = 0.1
    simplicity_weight = 1.0
    constraint_rho = 1.0
    steps = 1000

    for step in range(steps):
        if step % 10 == 0:
            # Discrete change.
            rewrites = path.propose_rewrites()
            velocity_constraint, collision_constraint, objective = compute_objectives(
                path, map, agent_radius
            )
            path_cost = (
                velocity_constraint.sum() * constraint_rho
                + collision_constraint.sum() * constraint_rho
                + objective
                + path.compute_simplicity_objective() * simplicity_weight
            )
            candidates = []

            print(collision_constraint.sum().item())

            for rewrite in rewrites:
                candidate = path.apply_rewrite(rewrite)
                grad = compute_grad(candidate, map, agent_radius, constraint_rho)

                velocity_constraint, collision_constraint, objective = (
                    compute_objectives(
                        candidate.apply_gradient(grad, learning_rate), map, agent_radius
                    )
                )
                candidate_cost = (
                    velocity_constraint.sum() * constraint_rho
                    + collision_constraint.sum() * constraint_rho
                    + objective
                    + candidate.compute_simplicity_objective() * simplicity_weight
                )
                candidates.append((candidate, candidate_cost))

            if candidates:
                best_candidate, best_cost = min(candidates, key=lambda x: x[1])
                if best_cost < path_cost:
                    path = best_candidate

        # Continuous change.
        grad = compute_grad(path, map, agent_radius, constraint_rho)
        grad = grad + torch.randn_like(grad) * 0.01

        if step % 10 == 0:
            render(
                map,
                [path],
                ["blue"],
                [grad],
                pause_behavior="pause" if (step + 10) < steps else "show",
            )

        path = path.apply_gradient(grad, learning_rate)
        constraint_rho = min(10.0, constraint_rho + 0.03)


if __name__ == "__main__":
    main()
