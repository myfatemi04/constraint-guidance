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

    def compute_min_distance_objective(self):
        return (self.vertices[1:, :2] - self.vertices[:-1, :2]).norm(dim=-1).sum()

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

    def apply_gradient(
        self, grad: torch.Tensor, learning_rate: float, max_velocity: float
    ):
        middle_vertices = self.vertices[1:-1] - (learning_rate * grad[1:-1])
        new_vertices = torch.cat(
            [self.vertices[:1], middle_vertices, self.vertices[-1:]], dim=0
        ).detach()

        # Project to feasible region for delta T.
        distances = torch.norm(new_vertices[1:] - new_vertices[:-1], dim=-1)
        min_dt = distances / max_velocity
        new_vertices[:-1, 2] = torch.maximum(new_vertices[:-1, 2], min_dt)

        # new_vertices[:, 2] = torch.maximum(
        #     new_vertices[:, 2], torch.zeros_like(new_vertices[:, 2])
        # )
        return Path(new_vertices.detach())


class Map:
    def __init__(self, obstacles: torch.Tensor):
        # (x, y, radius)
        self.obstacles = obstacles

    def compute_collision_constraint(self, path: Path, agent_radius: float):
        constraints = []
        for i in range(len(path.vertices) - 1):
            start = path.vertices[i][:2]
            end = path.vertices[i + 1][:2]
            dt = path.vertices[i][2]
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
                constraints.append(constraint * dt)
        return torch.stack(constraints).view(
            len(path.vertices) - 1, len(self.obstacles)
        )


def render(
    map: Map,
    paths: list[Path],
    colors: list[str],
    path_grads: list[torch.Tensor | None] | None = None,
    pause_behavior: Literal["show", "pause"] = "show",
    agent_radius: float = 0.2,
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
            plt.gca().add_patch(
                patches.Circle(
                    (vertex[0].item(), vertex[1].item()),
                    agent_radius,
                    color=colors[i],
                    alpha=0.5,
                )
            )

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


def compute_objectives(path: Path, map: Map, agent_radius: float, max_velocity: float):
    velocity_constraint = path.compute_velocity_constraint(max_velocity)
    collision_constraint = map.compute_collision_constraint(path, agent_radius)
    continuous_objective = path.compute_min_distance_objective()
    discrete_objective = path.compute_simplicity_objective()
    return (
        velocity_constraint,
        collision_constraint,
        continuous_objective,
        discrete_objective,
    )


def compute_grad(path: Path, map: Map, agent_radius: float, max_velocity: float):
    path.vertices = path.vertices.detach().requires_grad_()

    (
        velocity_constraint,
        collision_constraint,
        continuous_objective,
        discrete_objective,
    ) = compute_objectives(path, map, agent_radius, max_velocity)
    velocity_constraint.sum().backward(retain_graph=True)
    velocity_constraint_grad = path.vertices.grad  # type: ignore
    assert velocity_constraint_grad is not None
    path.vertices.grad = None

    collision_constraint.sum().backward(retain_graph=True)
    collision_constraint_grad = path.vertices.grad  # type: ignore
    assert collision_constraint_grad is not None
    path.vertices.grad = None

    continuous_objective.backward(retain_graph=True)
    continuous_objective_grad = path.vertices.grad  # type: ignore
    assert continuous_objective_grad is not None
    path.vertices.grad = None

    return (
        velocity_constraint_grad,
        collision_constraint_grad,
        continuous_objective_grad,
    )


def main():
    # Start with a simple path, going from (0, 0) to (0, 10) in 5 seconds. Maximum velocity is 2 m/s.
    path = Path(torch.tensor([[0, 0, 5], [0, 10, 0]], dtype=torch.float32))
    map_A = torch.tensor(
        [
            [0.2, 5.0, 0.5],
            [-0.2, 8.0, 0.5],
            [-0.8, 2.4, 2.0],
            [0.6, 7.0, 1.2],
        ],
        dtype=torch.float32,
    )
    map_B = torch.tensor(
        [
            [0.2, 5.0, 0.5],
            [-0.8, 2.4, 2.0],
            [0.6, 7.0, 1.2],
        ],
        dtype=torch.float32,
    )
    map_C = torch.tensor(
        [
            # [0.2, 5.0, 0.5],
            [-0.2, 8.0, 0.5],
            [-0.8, 3.0, 2.0],
            [0.6, 7.0, 1.2],
        ],
        dtype=torch.float32,
    )
    map = Map(map_C)
    agent_radius = 0.2
    learning_rate = 0.1
    simplicity_weight = 0.1
    constraint_rho = 0.1
    max_velocity = 2.0
    steps = 1000
    temperature = 10.0
    render_freq = 100

    for step in range(steps):
        if step % 100 == 0:
            # Discrete change.
            rewrites = path.propose_rewrites()
            (
                velocity_constraint,
                collision_constraint,
                continuous_objective,
                discrete_objective,
            ) = compute_objectives(path, map, agent_radius, max_velocity)
            path_cost = (
                velocity_constraint.sum() * constraint_rho
                + collision_constraint.sum() * constraint_rho
                + continuous_objective
                + path.compute_simplicity_objective() * simplicity_weight
            )
            candidates = []

            for rewrite in rewrites:
                candidate = path.apply_rewrite(rewrite)

                locally_optimized = candidate
                for _ in range(100):
                    (
                        velocity_constraint_grad,
                        collision_constraint_grad,
                        objective_grad,
                    ) = compute_grad(locally_optimized, map, agent_radius, max_velocity)

                    candidate_grad = (
                        velocity_constraint_grad * 0 + collision_constraint_grad
                    ) * constraint_rho + objective_grad

                    locally_optimized = locally_optimized.apply_gradient(
                        candidate_grad, learning_rate, max_velocity
                    )
                    (
                        candidate_velocity_constraint,
                        candidate_collision_constraint,
                        candidate_objective,
                        candidate_discrete_objective,
                    ) = compute_objectives(
                        locally_optimized, map, agent_radius, max_velocity
                    )

                    candidate_cost = (
                        candidate_velocity_constraint.sum() * 0
                        + candidate_collision_constraint.sum() * constraint_rho
                        + candidate_objective
                        + candidate.compute_simplicity_objective() * simplicity_weight
                    )

                    # print(f"Candidate cost: {candidate_cost.item()}")
                candidates.append((candidate, candidate_cost))

                print("Candidate costs:")
                print(
                    f"Velocity: {candidate_velocity_constraint.sum().item()} vs. {velocity_constraint.sum().item()}"
                )
                print(
                    f"Collision: {candidate_collision_constraint.sum().item()} vs. {collision_constraint.sum().item()}"
                )
                print(
                    f"Objective: {candidate_objective.item()} vs. {continuous_objective.item()}"
                )
                print(
                    f"Simplicity: {candidate.compute_simplicity_objective()} vs. {path.compute_simplicity_objective()}"
                )
                print(f"Total: {candidate_cost.item()} vs. {path_cost.item()}")

            if candidates:
                best_candidate, best_cost = min(candidates, key=lambda x: x[1])
                # if len(candidates) == 1:
                #     print(path_cost - best_cost)
                if best_cost < path_cost:
                    #      or torch.exp(
                    #     (path_cost - best_cost) / temperature
                    # ) > torch.rand(1):
                    path = best_candidate

        # Continuous change.
        velocity_constraint_grad, collision_constraint_grad, objective_grad = (
            compute_grad(path, map, agent_radius, max_velocity)
        )
        grad = (
            velocity_constraint_grad * 0 + collision_constraint_grad
        ) * constraint_rho + objective_grad
        # grad = grad + torch.randn_like(grad) * 1.0
        # grad = grad / torch.max(torch.norm(grad), torch.tensor(1.0))

        # Print out each constraint.
        # print(f"Velocity constraint: {velocity_constraint.tolist()}")
        # print(f"Collision constraint: {collision_constraint.tolist()}")
        # print(f"Objective: {objective.item()}")
        # print(f"Simplicity objective: {path.compute_simplicity_objective()}")
        # print(
        #     f"Velocity constraint grad norm: {torch.norm(velocity_constraint_grad).item()}"
        # )
        # print(
        #     f"Collision constraint grad norm: {torch.norm(collision_constraint_grad).item()}"
        # )
        # print(f"Objective grad norm: {torch.norm(objective_grad).item()}")

        if step % render_freq == 0:
            render(
                map,
                [
                    path,
                    candidates[-1][0].apply_gradient(
                        candidate_grad, learning_rate, max_velocity
                    ),
                ],
                ["blue", "green"],
                [grad, candidate_grad],
                pause_behavior="pause" if (step + render_freq) < steps else "show",
                agent_radius=agent_radius,
            )

            # render(
            #     map,
            #     [path],
            #     ["blue"],
            #     [grad],
            #     pause_behavior="pause" if (step + 10) < steps else "show",
            # )

        path = path.apply_gradient(grad, learning_rate, max_velocity)
        constraint_rho = min(5.0, constraint_rho + 0.03)
        temperature = max(0.1, temperature * 0.99)


if __name__ == "__main__":
    main()
