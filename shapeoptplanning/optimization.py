import torch


def squared_distance_to_line_segment(x1, x2, y):
    s = torch.clamp(
        ((y - x1) * (x2 - x1)).sum(dim=1) / ((x2 - x1) ** 2).sum(dim=1), 0, 1
    )
    xs = x1 + s * (x2 - x1)
    return (xs - y).pow(2).sum(dim=1)


class Path:
    def __init__(self, vertices: torch.Tensor, agent_radius: float):
        # list of (x, y, delta_t)
        self.vertices = vertices
        self.agent_radius = agent_radius

    def compute_velocity_constraint(self, max_velocity: float):
        dx_square = self.vertices[1:, :2] - self.vertices[:-1, :2]
        dt = self.vertices[:-1, 2]
        speed_squared = (dx_square**2).sum(dim=1) / dt**2
        return torch.maximum(
            speed_squared - max_velocity**2, torch.zeros_like(speed_squared)
        )

    def compute_min_time_objective(self):
        return self.vertices[:, 2].sum()

    def can_remove(self, i: int, theta_thres, d_thres):
        prev_vector = self.vertices[i] - self.vertices[i - 1]
        next_vector = self.vertices[i + 1] - self.vertices[i]
        dot = (prev_vector * next_vector).sum(dim=1)
        norm = torch.norm(prev_vector) * torch.norm(next_vector)
        cos_theta = dot / norm
        distance = squared_distance_to_line_segment(
            self.vertices[i - 1][:2], self.vertices[i + 1][:2], self.vertices[i][:2]
        )
        return cos_theta < torch.cos(theta_thres) and distance < d_thres**2

    def remove(self, i: int):
        # assumes can_remove check already passed
        before = self.vertices[:i]
        before[i - 1][2] += self.vertices[i][2]
        after = self.vertices[i + 1 :]
        new_vertices = torch.cat([before, after], dim=0)
        return Path(new_vertices, self.agent_radius)

    def insert(self, i: int):
        before = self.vertices[:i]
        after = self.vertices[i + 1 :]
        delta_t = self.vertices[i][2] / 2
        new_vertices = torch.cat(
            [
                before,
                torch.tensor([self.vertices[i][0], self.vertices[i][1], delta_t]),
                torch.tensor(
                    [
                        (self.vertices[i][0] + self.vertices[i + 1][0]) / 2,
                        (self.vertices[i][1] + self.vertices[i + 1][1]) / 2,
                        delta_t,
                    ]
                ),
                after,
            ],
            dim=0,
        )
        return Path(new_vertices, self.agent_radius)


class Map:
    def __init__(self, obstacles: torch.Tensor):
        # (x, y, radius)
        self.obstacles = obstacles

    def compute_collision_constraint(self, path: Path):
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
                constraints.append(
                    torch.maximum(
                        (obstacle_radius + path.agent_radius) ** 2 - squared_distance,
                        torch.zeros_like(squared_distance),
                    )
                )
        return torch.stack(constraints).view(
            len(path.vertices) - 1, len(self.obstacles)
        )
