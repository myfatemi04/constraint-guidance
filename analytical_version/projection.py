import time
from dataclasses import dataclass
from typing import Any, TypeVar

import matplotlib.patches as patches

# Visualize trajectory (and dynamic obstacles)
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,  # type: ignore
    Objective,
    RangeSet,
    Set,
    SolverFactory,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

ArrayType = TypeVar("ArrayType", np.ndarray, torch.Tensor, Var)


@dataclass
class Obstacles:
    positions: np.ndarray
    radii: np.ndarray


def circle_signed_distance(
    circle_1_x,
    circle_1_y,
    circle_1_radius,
    circle_2_x,
    circle_2_y,
    circle_2_radius,
):
    center_distance = (circle_1_x - circle_2_x) ** 2 + (circle_1_y - circle_2_y) ** 2
    min_allowed_distance = (circle_1_radius + circle_2_radius) ** 2
    return center_distance - min_allowed_distance


def _maybe_value(v):
    try:
        return value(v)
    except ValueError:
        return -100.0


def dictionary_to_list(d: dict, shape) -> list:
    array = np.zeros(shape).tolist()
    for idxs, v in d.items():
        entry = array
        for idx in idxs[:-1]:
            entry = entry[idx]  # type: ignore
        entry[idxs[-1]] = v  # type: ignore
    return array  # type: ignore


@dataclass
class SolutionValue[ArrayType]:
    agent_agent_distances: ArrayType
    agent_obstacle_distances: ArrayType
    agent_positions: ArrayType

    @classmethod
    def get_symbolic_from_model(cls, model) -> "SolutionValue":
        return SolutionValue(
            agent_positions=model.AgentPositions,
            agent_agent_distances=model.AgentAgentDistances,
            agent_obstacle_distances=model.AgentObstacleDistances,
        )

    @classmethod
    def get_value_from_model(cls, model) -> "SolutionValue[np.ndarray]":
        agent_positions = np.array(
            dictionary_to_list(
                {k: value(model.AgentPositions[k]) for k in model.AgentPositions},
                shape=(
                    len(model.Time),
                    len(model.Agent),
                    len(model.Dim),
                ),
            )
        )
        agent_agent_distances = np.array(
            dictionary_to_list(
                {
                    k: _maybe_value(model.AgentAgentDistances[k])
                    for k in model.AgentAgentDistances
                },
                shape=(
                    len(model.Time),
                    len(model.Agent),
                    len(model.Agent),
                ),
            )
        )
        agent_obstacle_distances = np.array(
            dictionary_to_list(
                {
                    k: _maybe_value(model.AgentObstacleDistances[k])
                    for k in model.AgentObstacleDistances
                },
                shape=(
                    len(model.Time),
                    len(model.Agent),
                    len(model.Obstacle),
                ),
            )
        )
        return SolutionValue(
            agent_positions=agent_positions,
            agent_agent_distances=agent_agent_distances,
            agent_obstacle_distances=agent_obstacle_distances,
        )


@dataclass
class SolutionResult:
    sol: SolutionValue[np.ndarray]
    status: SolverStatus
    termination_condition: TerminationCondition
    duration: float

    @property
    def solved_optimally(self):
        return (self.status == SolverStatus.ok) and (
            self.termination_condition == TerminationCondition.optimal
        )


@dataclass
class ALMResult:
    step_results: list[SolutionResult]
    agent_agent_nonpenetration_residual_history: list[np.ndarray]
    agent_obstacle_nonpenetration_residual_history: list[np.ndarray]


@dataclass
class InequalityConstraintResult:
    title: str
    value: float
    minimum_value: float

    @property
    def implied_slack(self):
        """Must be nonpositive to be feasible."""
        return self.value - self.minimum_value


@dataclass
class EqualityConstraintResult:
    title: str
    value: float
    equality_value: float

    @property
    def implied_slack(self):
        """Must be zero to be feasible."""
        return self.value - self.equality_value


@dataclass
class ConstraintEvaluationResult:
    agent_agent_penetration: dict[tuple[int, int, int], InequalityConstraintResult]
    """ time_step, (source_agent_index, target_agent_index) """
    agent_obstacle_penetration: dict[tuple[int, int, int], InequalityConstraintResult]

    """ time_step, agent_index, obstacle_index """
    speed: dict[tuple[int, int], InequalityConstraintResult]
    """ time_step, agent_index """

    agent_start_positions: dict[tuple[int, int], EqualityConstraintResult]
    agent_end_positions: dict[tuple[int, int], EqualityConstraintResult]

    def get_pyomo_speed_constraint(self, model: ConcreteModel) -> Constraint:
        return Constraint(
            model.Time,
            model.Agent,
            rule=lambda _model, time_index, agent_index: (
                self.speed[time_index, agent_index].value
                >= self.speed[time_index, agent_index].minimum_value
                if (time_index < model.Time.last())  # type: ignore
                else Constraint.Skip
            ),
        )

    def get_pyomo_start_end_position_constraints(
        self, model: ConcreteModel
    ) -> tuple[Constraint, Constraint]:
        return Constraint(
            model.Agent,
            model.Dim,
            rule=lambda _model, agent_index, dim: (
                self.agent_start_positions[agent_index, dim].implied_slack == 0
            ),
        ), Constraint(
            model.Agent,
            model.Dim,
            rule=lambda _model, agent_index, dim: (
                self.agent_end_positions[agent_index, dim].implied_slack == 0
            ),
        )

    def compute_penetration_alm_objective(
        self,
        nu_agent_agent_penetration: np.ndarray,
        nu_agent_obstacle_penetration: np.ndarray,
        agent_agent_penetration_slack,
        agent_obstacle_penetration_slack,
        rho: float,
    ) -> tuple[float, dict, dict]:
        """
        Constraints are created by the NonNegativeReals in the agent_agent_penetration_variable results.
        """
        agent_agent_constraint_residual = {
            k: agent_agent_penetration_slack[k]
            - self.agent_agent_penetration[k].implied_slack
            for k in self.agent_agent_penetration.keys()
        }
        agent_obstacle_constraint_residual = {
            k: agent_obstacle_penetration_slack[k]
            - self.agent_obstacle_penetration[k].implied_slack
            for k in self.agent_obstacle_penetration.keys()
        }
        alm_objective = sum(
            agent_agent_constraint_residual[k] * nu_agent_agent_penetration[k]
            + 0.5 * rho * agent_agent_constraint_residual[k] ** 2
            for k in self.agent_agent_penetration.keys()
        ) + sum(
            agent_obstacle_constraint_residual[k] * nu_agent_obstacle_penetration[k]
            + 0.5 * rho * agent_obstacle_constraint_residual[k] ** 2
            for k in self.agent_obstacle_penetration.keys()
        )
        return (
            alm_objective,
            agent_agent_constraint_residual,
            agent_obstacle_constraint_residual,
        )


@dataclass
class Problem:
    num_timesteps: int
    agent_start_positions: np.ndarray
    agent_end_positions: np.ndarray
    agent_radii: np.ndarray
    agent_max_speeds: np.ndarray
    obstacle_positions: np.ndarray
    obstacle_radii: np.ndarray

    def convert_torch(self):
        self.agent_start_positions = torch.from_numpy(self.agent_start_positions)  # type: ignore
        self.agent_end_positions = torch.from_numpy(self.agent_end_positions)  # type: ignore
        self.agent_radii = torch.from_numpy(self.agent_radii)  # type: ignore
        self.agent_max_speeds = torch.from_numpy(self.agent_max_speeds)  # type: ignore
        self.obstacle_positions = torch.from_numpy(self.obstacle_positions)  # type: ignore
        self.obstacle_radii = torch.from_numpy(self.obstacle_radii)  # type: ignore

    @property
    def num_agents(self):
        return self.agent_start_positions.shape[0]

    @property
    def num_obstacles(self):
        return self.obstacle_positions.shape[0]

    def evaluate_agent_agent_nonpenetration_constraint(
        self, sol: SolutionValue
    ) -> dict[tuple[int, int, int], InequalityConstraintResult]:
        return {
            (
                time_index,
                source_agent_index,
                target_agent_index,
            ): InequalityConstraintResult(
                title=f"agent_agent_nonpenetration:{time_index}:{source_agent_index}:{target_agent_index}",
                value=circle_signed_distance(
                    circle_1_x=sol.agent_positions[time_index, source_agent_index, 0],
                    circle_1_y=sol.agent_positions[time_index, source_agent_index, 1],
                    circle_1_radius=self.agent_radii[source_agent_index],
                    circle_2_x=sol.agent_positions[time_index, target_agent_index, 0],
                    circle_2_y=sol.agent_positions[time_index, target_agent_index, 1],
                    circle_2_radius=self.agent_radii[target_agent_index],
                ),
                minimum_value=0.0,
            )
            for source_agent_index in range(self.num_agents)
            for target_agent_index in range(self.num_agents)
            for time_index in range(self.num_timesteps)
            if target_agent_index > source_agent_index
        }

    def evaluate_speed_constraint(
        self, sol: SolutionValue
    ) -> dict[tuple[int, int], InequalityConstraintResult]:
        return {
            (time_index, agent_index): InequalityConstraintResult(
                title=f"speed_constraint:{time_index}:{agent_index}",
                value=-sum(
                    [
                        (
                            sol.agent_positions[time_index + 1, agent_index, k]
                            - sol.agent_positions[time_index, agent_index, k]
                        )
                        ** 2
                        for k in [0, 1]
                    ]
                ),
                minimum_value=-(self.agent_max_speeds[agent_index] ** 2),
            )
            for agent_index in range(self.num_agents)
            for time_index in range(self.num_timesteps - 1)
        }

    def evaluate_agent_obstacle_nonpenetration_constraint(
        self, sol: SolutionValue
    ) -> dict[tuple[int, int, int], InequalityConstraintResult]:
        return {
            (
                time_index,
                agent_index,
                obs_index,
            ): InequalityConstraintResult(
                title=f"agent_obstacle_nonpenetration:{time_index}:{agent_index}:{obs_index}",
                value=circle_signed_distance(
                    circle_1_x=sol.agent_positions[time_index, agent_index, 0],
                    circle_1_y=sol.agent_positions[time_index, agent_index, 1],
                    circle_1_radius=self.agent_radii[agent_index],
                    circle_2_x=self.obstacle_positions[obs_index, 0],
                    circle_2_y=self.obstacle_positions[obs_index, 1],
                    circle_2_radius=self.obstacle_radii[obs_index],
                ),
                minimum_value=0.0,
            )
            for obs_index in range(self.num_obstacles)
            for agent_index in range(self.num_agents)
            for time_index in range(self.num_timesteps)
        }

    def get_constraints(self, sol: SolutionValue):
        return ConstraintEvaluationResult(
            agent_agent_penetration=self.evaluate_agent_agent_nonpenetration_constraint(
                sol
            ),
            agent_obstacle_penetration=self.evaluate_agent_obstacle_nonpenetration_constraint(
                sol
            ),
            speed=self.evaluate_speed_constraint(sol),
            agent_start_positions={
                (i, d): EqualityConstraintResult(
                    title=f"agent_start_pos:{i}:{d}",
                    value=sol.agent_positions[0, i, d],
                    equality_value=self.agent_start_positions[i, d],
                )
                for i in range(self.num_agents)
                for d in range(2)
            },
            agent_end_positions={
                (i, d): EqualityConstraintResult(
                    title=f"agent_end_pos:{i}:{d}",
                    value=sol.agent_positions[self.num_timesteps - 1, i, d],
                    equality_value=self.agent_end_positions[i, d],
                )
                for i in range(self.num_agents)
                for d in range(2)
            },
        )

    def evaluate_distance_objective(self, sol: SolutionValue):
        return sum(
            (
                sol.agent_positions[time_index + 1, agent_index, k]
                - sol.agent_positions[time_index, agent_index, k]
            )
            ** 2
            for time_index in range(self.num_timesteps - 1)
            for agent_index in range(self.num_agents)
            for k in [0, 1]
        )

    def visualize(self, sol: SolutionValue, ax):
        # Plot the obstacles
        for obs_index in range(self.num_obstacles):
            ax.add_patch(
                patches.Circle(
                    (
                        self.obstacle_positions[obs_index, 0],
                        self.obstacle_positions[obs_index, 1],
                    ),
                    self.obstacle_radii[obs_index],
                    color="r",
                    alpha=0.5,
                )
            )

        agent_pos = sol.agent_positions
        if isinstance(agent_pos, torch.Tensor):
            agent_pos = agent_pos.detach().cpu().numpy()

        # Plot the agents' trajectories
        for agent_index in range(self.num_agents):
            if agent_pos.shape[0] == 1:
                ax.add_patch(
                    patches.Circle(
                        (agent_pos[0, agent_index, 0], agent_pos[0, agent_index, 1]),
                        self.agent_radii[agent_index],
                    )
                )
            else:
                ax.plot(
                    agent_pos[:, agent_index, 0],
                    agent_pos[:, agent_index, 1],
                    marker="o",
                    label=f"Agent {agent_index}",
                    # set size to agent radius
                    markersize=self.agent_radii[agent_index] * 10,
                )
        # Plot the agents' start and goal positions
        for agent_index in range(self.num_agents):
            start_pos = self.agent_start_positions[agent_index]
            goal_pos = self.agent_end_positions[agent_index]
            ax.plot(
                start_pos[0],
                start_pos[1],
                marker="o",
                color="green",
                markersize=10,
                label=f"Start {agent_index}",
            )
            ax.plot(
                goal_pos[0],
                goal_pos[1],
                marker="*",
                color="blue",
                markersize=10,
                label=f"Goal {agent_index}",
            )

        ax.set_aspect("equal")

    @classmethod
    def from_json(cls, entry):
        return cls(
            num_timesteps=entry["num_timesteps"],
            agent_start_positions=np.array(entry["agents"]["start_positions"]),
            agent_end_positions=np.array(entry["agents"]["end_positions"]),
            agent_radii=np.array(entry["agents"]["radii"]),
            agent_max_speeds=np.array(entry["agents"]["max_speeds"]),
            obstacle_positions=np.array(entry["obstacles"]["positions"]),
            obstacle_radii=np.array(entry["obstacles"]["radii"]),
        )


def solve_alm_subproblem(
    problem: Problem,
    nu_agent_agent_nonpenetration,
    nu_agent_obstacle_nonpenetration,
    rho,
) -> SolutionResult:
    model: Any = ConcreteModel()

    # Define sets
    agent_collision_indices = [
        (time_index, source_agent_index, target_agent_index)
        for time_index in range(problem.num_timesteps)
        for source_agent_index in range(problem.num_agents)
        for target_agent_index in range(problem.num_agents)
        if target_agent_index > source_agent_index
    ]
    model.Time = RangeSet(0, problem.num_timesteps - 1)
    model.Agent = RangeSet(0, problem.num_agents - 1)
    model.Obstacle = RangeSet(0, problem.num_obstacles - 1)
    model.Dim = RangeSet(0, 1)
    model.AgentCollisionIndices = Set(initialize=agent_collision_indices)
    model.AgentPositions = Var(model.Time, model.Agent, model.Dim, bounds=(-1, 1))
    model.AgentObstacleDistances = Var(
        model.Time,
        model.Agent,
        model.Obstacle,
        within=NonNegativeReals,
    )
    model.AgentAgentDistances = Var(
        model.AgentCollisionIndices,
        within=NonNegativeReals,
    )
    symbolic_solution_value = SolutionValue.get_symbolic_from_model(model)
    symbolic_constraints = problem.get_constraints(symbolic_solution_value)
    model.StartPositionConstraint, model.EndPositionConstraint = (
        symbolic_constraints.get_pyomo_start_end_position_constraints(model)
    )
    model.SpeedConstraint = symbolic_constraints.get_pyomo_speed_constraint(model)
    model.objective = Objective(
        rule=lambda _model: (
            problem.evaluate_distance_objective(symbolic_solution_value)
            + symbolic_constraints.compute_penetration_alm_objective(
                nu_agent_agent_nonpenetration,
                nu_agent_obstacle_nonpenetration,
                _model.AgentAgentDistances,
                _model.AgentObstacleDistances,
                rho,
            )[0]
        ),
        sense=minimize,
    )

    # Solve the model
    t0 = time.time()
    solver = SolverFactory("ipopt")
    results = solver.solve(model, tee=False)
    t1 = time.time()

    return SolutionResult(
        sol=SolutionValue.get_value_from_model(model),
        status=results.solver.status,
        termination_condition=results.solver.termination_condition,
        duration=t1 - t0,
    )


def solve_alm(
    problem: Problem,
    rho: float,
    rho_factor: float,
    num_alm_iterations: int,
    tolerance: float,
) -> ALMResult:
    """
    x: tensor of shape (batch_size, horizon, state_dim)
    """

    nu_agent_agent_nonpenetration = np.zeros(
        (problem.num_timesteps, problem.num_agents, problem.num_agents)
    )
    nu_agent_obstacle_nonpenetration = np.zeros(
        (problem.num_timesteps, problem.num_agents, problem.num_obstacles)
    )

    step_results: list[SolutionResult] = []
    agent_agent_nonpenetration_residual_history: list[np.ndarray] = []
    agent_obstacle_nonpenetration_residual_history: list[np.ndarray] = []

    t_alm_start = time.time()
    for steps in range(num_alm_iterations):
        t_alm_step_start = time.time()
        result = solve_alm_subproblem(
            problem,
            nu_agent_agent_nonpenetration,
            nu_agent_obstacle_nonpenetration,
            rho,
        )

        # TODO: Check if solver failed.
        agent_agent_nonpenetration_residuals = (
            problem.evaluate_agent_agent_nonpenetration_constraint(result.sol)
        )
        agent_obstacle_nonpenetration_residuals = (
            problem.evaluate_agent_obstacle_nonpenetration_constraint(result.sol)
        )
        agent_agent_nonpenetration_infeasible_count = sum(
            int(v.implied_slack <= -tolerance)
            for v in agent_agent_nonpenetration_residuals.values()
        )
        agent_obstacle_nonpenetration_infeasible_count = sum(
            int(v.implied_slack <= -tolerance)
            for v in agent_obstacle_nonpenetration_residuals.values()
        )
        agent_agent_nonpenetration_residuals_array = np.array(
            dictionary_to_list(
                {
                    k: agent_agent_nonpenetration_residuals[k].implied_slack
                    for k in agent_agent_nonpenetration_residuals.keys()
                },
                shape=(problem.num_timesteps, problem.num_agents, problem.num_agents),
            )
        )
        agent_obstacle_nonpenetration_residuals_array = np.array(
            dictionary_to_list(
                {
                    k: agent_obstacle_nonpenetration_residuals[k].implied_slack
                    for k in agent_obstacle_nonpenetration_residuals.keys()
                },
                shape=(
                    problem.num_timesteps,
                    problem.num_agents,
                    problem.num_obstacles,
                ),
            )
        )
        agent_obstacle_residual_norm = np.linalg.norm(
            agent_obstacle_nonpenetration_residuals_array
        )
        agent_agent_residual_norm = np.linalg.norm(
            agent_agent_nonpenetration_residuals_array
        )

        logger.info(
            f"ALM step {steps}: {agent_obstacle_residual_norm=}, {agent_agent_residual_norm=} {agent_obstacle_nonpenetration_infeasible_count=} {agent_agent_nonpenetration_infeasible_count=}"
        )

        agent_agent_nonpenetration_residual_history.append(
            agent_agent_nonpenetration_residuals_array
        )
        agent_obstacle_nonpenetration_residual_history.append(
            agent_obstacle_nonpenetration_residuals_array
        )
        step_results.append(result)

        feasible = (
            agent_agent_nonpenetration_infeasible_count
            + agent_obstacle_nonpenetration_infeasible_count
            < 1
        )
        if feasible:
            break

        # update the lagrange parameters
        rho = rho * rho_factor
        nu_agent_obstacle_nonpenetration = (
            nu_agent_obstacle_nonpenetration
            + rho * agent_obstacle_nonpenetration_residuals_array
        )
        nu_agent_agent_nonpenetration = (
            nu_agent_agent_nonpenetration
            + rho * agent_agent_nonpenetration_residuals_array
        )
        t_alm_step_end = time.time()

        logger.info(
            f"ALM step {steps} completed in {t_alm_step_end - t_alm_step_start} seconds."
        )

    t_alm_end = time.time()
    logger.info(f"Total time: {t_alm_end - t_alm_start} seconds.")

    return ALMResult(
        step_results=step_results,
        agent_agent_nonpenetration_residual_history=agent_agent_nonpenetration_residual_history,
        agent_obstacle_nonpenetration_residual_history=agent_obstacle_nonpenetration_residual_history,
    )


def _convert_data_format(data):
    entries = []
    for sample_idx in range(len(data)):
        for group_idx in range(len(data[sample_idx])):
            obstacles_info, agents_info = data[sample_idx][group_idx]
            obstacle_positions = np.array([o[0] for o in obstacles_info])
            obstacle_radii = np.array([o[1] for o in obstacles_info])
            agent_start_positions = np.array([a[0] for a in agents_info])
            agent_end_positions = np.array([a[1] for a in agents_info])
            agent_radii = np.array([a[2] for a in agents_info])
            entries.append(
                {
                    "num_timesteps": 64,
                    "sample_idx": sample_idx,
                    "agents": {
                        "start_positions": agent_start_positions.tolist(),
                        "end_positions": agent_end_positions.tolist(),
                        "radii": agent_radii.tolist(),
                        "max_speeds": (0.05 * np.ones_like(agent_radii)).tolist(),
                    },
                    "obstacles": {
                        "positions": obstacle_positions.tolist(),
                        "radii": obstacle_radii.tolist(),
                    },
                }
            )

    return entries


def _convert():
    import pickle
    import pathlib
    import json

    instances = pathlib.Path("instances_data/")

    for file in instances.iterdir():
        if file.suffix == ".pkl":
            # sample_idx, group_idx, (obstacles_info, agents_info)
            with open(file, "rb") as f:
                data = pickle.load(f)

            entries = _convert_data_format(data)

            with open(f"instances_data/{file.stem}.json", "w") as f:
                json.dump(entries, f)


def main():
    import json
    import tqdm

    with open("instances_data/instances_dense.json", "r") as f:
        data = json.load(f)

    # find problem with 9 robots
    # prob = [d for d in data if len(d["agents"]["start_positions"]) == 9][0]
    prob = [
        d
        for d in data
        if d["sample_idx"] == 4 and len(d["agents"]["start_positions"]) == 9
    ][0]

    problem = Problem.from_json(prob)
    sol = SolutionValue(
        agent_agent_distances=torch.zeros((65, problem.num_agents, problem.num_agents)),
        agent_obstacle_distances=torch.zeros(
            (65, problem.num_agents, problem.num_obstacles)
        ),
        agent_positions=torch.randn((65, problem.num_agents, 2), requires_grad=True),
    )

    opt = torch.optim.Adam([sol.agent_positions], lr=0.1)

    obs_poses = torch.from_numpy(problem.obstacle_positions)
    agent_radii = torch.from_numpy(problem.agent_radii)
    obstacle_radii = torch.from_numpy(problem.obstacle_radii)

    obstacle_penalty_weight = 50
    agent_penalty_weight = 50
    lowlevel_vel_penalty_weight = 10
    transition_by = 2000
    energy_lowlevel_weight = 0  # to 3
    energy_highlevel_weight = 3.0  # to 0
    rate = 1.005

    for i in tqdm.tqdm(range(8000)):
        plan_highlevel = sol.agent_positions[::4]
        highlevel_vel_sq = (
            (plan_highlevel[1:, :, :] - plan_highlevel[:-1, :, :]).pow(2).sum(-1)
        )
        energy_highlevel = highlevel_vel_sq.sum()
        lowlevel_vel_sq = (
            (sol.agent_positions[1:, :, :] - sol.agent_positions[:-1, :, :])
            .pow(2)
            .sum(-1)
        )
        energy_lowlevel = lowlevel_vel_sq.sum()

        highlevel_vel_penalty = (
            highlevel_vel_sq[highlevel_vel_sq > (0.05**2)] - (0.05**2)
        ).sum()
        lowlevel_vel_penalty = torch.relu(lowlevel_vel_sq - (0.05**2)).pow(2).sum()

        # (t, a, o)
        obstacle_center_dist_sq = (
            (sol.agent_positions.unsqueeze(2) - obs_poses.unsqueeze(0).unsqueeze(0))
            .pow(2)
            .sum(-1)
        )
        obstacle_signed_distances = obstacle_center_dist_sq - (
            agent_radii.view(1, -1, 1) + obstacle_radii.view(1, 1, -1)
        ).pow(2)
        obstacle_penalties = torch.relu(-obstacle_signed_distances).pow(2).sum()

        # (t, a, a)
        agent_center_dist_sq = (
            (sol.agent_positions.unsqueeze(2) - sol.agent_positions.unsqueeze(1))
            .pow(2)
            .sum(-1)
        )
        agent_signed_distances = agent_center_dist_sq - (
            agent_radii.view(1, -1, 1) + agent_radii.view(1, 1, -1)
        ).pow(2)
        agent_signed_distances = (
            agent_signed_distances
            # to ignore self-interactions in the relu
            + torch.eye(problem.num_agents).view(
                1, problem.num_agents, problem.num_agents
            )
            * 1e6
        )
        agent_penalties = torch.relu(-agent_signed_distances).pow(2).sum()

        loss = (
            energy_highlevel * energy_highlevel_weight
            + energy_lowlevel * energy_lowlevel_weight
            + obstacle_penalties * obstacle_penalty_weight
            + agent_penalties * agent_penalty_weight
            # + highlevel_vel_penalty * 10
            + lowlevel_vel_penalty * lowlevel_vel_penalty_weight
        )

        if (i + 1) % 10 == 0 and i < 2000:
            obstacle_penalty_weight *= rate
            agent_penalty_weight *= rate
            lowlevel_vel_penalty_weight *= rate

        energy_lowlevel_weight = 10 * min(i, transition_by) / transition_by
        energy_highlevel_weight = 10 - energy_lowlevel_weight

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sol.agent_positions[0, :, :] = torch.from_numpy(
                problem.agent_start_positions
            )
            sol.agent_positions[-1, :, :] = torch.from_numpy(
                problem.agent_end_positions
            )

    print("agent_penalties:", agent_penalties.item())
    print("obstacle_penalties:", obstacle_penalties.item())
    print("energy_lowlevel:", energy_lowlevel.item())
    plt.clf()
    problem.visualize(sol, plt.gca())
    plt.show()

    # animate the solution
    for i in range(problem.num_timesteps):
        plt.clf()
        problem.visualize(
            SolutionValue(
                agent_agent_distances=None,
                agent_obstacle_distances=None,
                agent_positions=sol.agent_positions[i : i + 1].detach().cpu().numpy(),
            ),
            plt.gca(),
        )
        plt.pause(0.1)


if __name__ == "__main__":
    main()
