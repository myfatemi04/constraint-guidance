import time
from dataclasses import dataclass
from typing import Any, TypeVar

import matplotlib.patches as patches

# Visualize trajectory (and dynamic obstacles)
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from pyomo.contrib.appsi.solvers.ipopt import Ipopt
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
            agent_obstacle_distances=model.AgentObjectDistances,
        )

    @classmethod
    def get_value_from_model(cls, model) -> "SolutionValue[np.ndarray]":
        agent_positions = np.array(
            [
                [
                    [
                        value(model.AgentPositions[time_index, agent_index, dim])
                        for dim in range(2)
                    ]
                    for agent_index in range(model.Agent.last() + 1)
                ]
                for time_index in range(model.Time.last() + 1)
            ]
        )
        agent_agent_distances = np.array(
            [
                [
                    [
                        value(
                            model.AgentAgentDistances[
                                time_index, (min(a1, a2), max(a1, a2))
                            ]
                        )
                        for a2 in range(model.Agent.last() + 1)
                    ]
                    for a1 in range(model.Agent.last() + 1)
                ]
                for time_index in range(model.Time.last() + 1)
            ]
        )
        agent_obstacle_distances = np.array(
            [
                [
                    [
                        value(
                            model.AgentObjectDistances[
                                time_index, agent_index, obs_index
                            ]
                        )
                        for obs_index in range(model.Obstacle.last() + 1)
                    ]
                    for agent_index in range(model.Agent.last() + 1)
                ]
                for time_index in range(model.Time.last() + 1)
            ]
        )
        return SolutionValue(
            agent_positions=agent_positions,
            agent_agent_distances=agent_agent_distances,
            agent_obstacle_distances=agent_obstacle_distances,
        )


@dataclass
class SolutionResult:
    value: SolutionValue[np.ndarray]
    status: SolverStatus
    termination_condition: TerminationCondition

    @property
    def solved_optimally(self):
        return (self.status == SolverStatus.ok) and (
            self.termination_condition == TerminationCondition.optimal
        )


@dataclass
class SingleDynamicConstraintEvaluationResult:
    title: str
    implied_value: float
    minimum_value: float


@dataclass
class FullDynamicConstraintEvaluationResult:
    agent_agent_penetration: list[
        dict[tuple[int, int], SingleDynamicConstraintEvaluationResult]
    ]
    """ time_step, (source_agent_index, target_agent_index) """
    agent_obstacle_penetration: list[
        list[list[SingleDynamicConstraintEvaluationResult]]
    ]
    """ time_step, agent_index, obstacle_index """
    speed: list[list[SingleDynamicConstraintEvaluationResult]]
    """ time_step, agent_index """

    @property
    def num_agents(self):
        return len(self.agent_obstacle_penetration[0])

    @property
    def num_timesteps(self):
        return len(self.speed)

    def get_speed_constraint(self, model: ConcreteModel) -> Constraint:
        return Constraint(
            model.Time,
            model.Agent,
            rule=lambda _model, time_index, agent_index: self.speed[time_index][
                agent_index
            ]
            if time_index < len(self.speed) - 1
            else Constraint.Skip,
        )

    def compute_penetration_alm_objective(
        self,
        nu_agent_agent_penetration: np.ndarray,
        nu_agent_obstacle_penetration: np.ndarray,
        agent_agent_penetration_variable,
        agent_obstacle_penetration_variable,
        rho: float,
    ):
        """
        Constraints are created by the NonNegativeReals in the agent_agent_penetration_variable results.
        Constraint residuals must be nonpositive to be feasible.
        """
        agent_agent_constraint_residual = [
            {
                (
                    source_agent_index,
                    target_agent_index,
                ): agent_agent_penetration_variable[time_index][source_agent_index][
                    target_agent_index
                ]
                - self.agent_agent_penetration[time_index][
                    source_agent_index, target_agent_index
                ].implied_value
                for (
                    source_agent_index,
                    target_agent_index,
                ) in self.agent_agent_penetration[time_index].keys()
            }
            for time_index in range(len(self.agent_agent_penetration))
        ]
        agent_obstacle_constraint_residual = [
            [
                [
                    agent_obstacle_penetration_variable[time_index][agent_index][
                        obstacle_index
                    ]
                    - self.agent_obstacle_penetration[time_index][agent_index][
                        obstacle_index
                    ].implied_value
                    for obstacle_index in range(
                        len(self.agent_obstacle_penetration[time_index][agent_index])
                    )
                ]
                for agent_index in range(
                    len(self.agent_obstacle_penetration[time_index])
                )
            ]
            for time_index in range(len(self.agent_obstacle_penetration))
        ]

        alm_objective = 0.0
        # Agent-agent penetration terms
        for time_index in range(len(self.agent_agent_penetration)):
            for (
                source_agent_index,
                target_agent_index,
            ) in self.agent_agent_penetration[time_index].keys():
                residual = agent_agent_constraint_residual[time_index][
                    (source_agent_index, target_agent_index)
                ]
                nu = nu_agent_agent_penetration[
                    time_index, source_agent_index, target_agent_index
                ]
                alm_objective += nu * residual + 0.5 * rho * residual**2
        # Agent-obstacle penetration terms
        for time_index in range(len(self.agent_obstacle_penetration)):
            for agent_index in range(len(self.agent_obstacle_penetration[time_index])):
                for obstacle_index in range(
                    len(self.agent_obstacle_penetration[time_index][agent_index])
                ):
                    residual = agent_obstacle_constraint_residual[time_index][
                        agent_index
                    ][obstacle_index]
                    nu = nu_agent_obstacle_penetration[
                        time_index, agent_index, obstacle_index
                    ]
                    alm_objective += nu * residual + 0.5 * rho * residual**2

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
    obstacles: Obstacles
    agent_radii: np.ndarray
    agent_max_speeds: np.ndarray

    @property
    def num_agents(self):
        return self.agent_start_positions.shape[0]

    @property
    def num_obstacles(self):
        return self.obstacles.positions.shape[0]

    def evaluate_agent_agent_nonpenetration_constraint(
        self, sol: SolutionValue
    ) -> list[dict[tuple[int, int], SingleDynamicConstraintEvaluationResult]]:
        return [
            {
                (
                    source_agent_index,
                    target_agent_index,
                ): SingleDynamicConstraintEvaluationResult(
                    title=f"agent_agent_nonpenetration:{time_index}:{source_agent_index}:{target_agent_index}",
                    implied_value=circle_signed_distance(
                        circle_1_x=sol.agent_positions[
                            time_index, source_agent_index, 0
                        ],
                        circle_1_y=sol.agent_positions[
                            time_index, source_agent_index, 1
                        ],
                        circle_1_radius=self.agent_radii[source_agent_index],
                        circle_2_x=sol.agent_positions[
                            time_index, target_agent_index, 0
                        ],
                        circle_2_y=sol.agent_positions[
                            time_index, target_agent_index, 1
                        ],
                        circle_2_radius=self.agent_radii[target_agent_index],
                    ),
                    minimum_value=0.0,
                )
                for source_agent_index in range(self.num_agents)
                for target_agent_index in range(self.num_agents)
                if target_agent_index > source_agent_index
            }
            for time_index in range(self.num_timesteps)
        ]

    def evaluate_speed_constraint(
        self, sol: SolutionValue
    ) -> list[list[SingleDynamicConstraintEvaluationResult]]:
        return [
            [
                SingleDynamicConstraintEvaluationResult(
                    title=f"speed_constraint:{time_index}:{agent_index}",
                    implied_value=sum(
                        [
                            (
                                sol.agent_positions[time_index + 1, agent_index, k]
                                - sol.agent_positions[time_index, agent_index, k]
                            )
                            ** 2
                            for k in [0, 1]
                        ]
                    ),
                    minimum_value=self.agent_max_speeds[agent_index] ** 2,
                )
                for agent_index in range(self.num_agents)
            ]
            for time_index in range(self.num_timesteps - 1)
        ]

    def evaluate_agent_obstacle_nonpenetration_constraint(
        self, sol: SolutionValue, obstacles: Obstacles
    ) -> list[list[list[SingleDynamicConstraintEvaluationResult]]]:
        return [
            [
                [
                    SingleDynamicConstraintEvaluationResult(
                        title=f"agent_obstacle_nonpenetration:{time_index}:{agent_index}:{obs_index}",
                        implied_value=circle_signed_distance(
                            circle_1_x=sol.agent_positions[time_index, agent_index, 0],
                            circle_1_y=sol.agent_positions[time_index, agent_index, 1],
                            circle_1_radius=self.agent_radii[agent_index],
                            circle_2_x=obstacles.positions[obs_index, 0],
                            circle_2_y=obstacles.positions[obs_index, 1],
                            circle_2_radius=obstacles.radii[obs_index],
                        ),
                        minimum_value=0.0,
                    )
                    for obs_index in range(self.num_obstacles)
                ]
                for agent_index in range(self.num_agents)
            ]
            for time_index in range(self.num_timesteps)
        ]

    def get_constraints(self, sol: SolutionValue):
        return FullDynamicConstraintEvaluationResult(
            agent_agent_penetration=self.evaluate_agent_agent_nonpenetration_constraint(
                sol
            ),
            agent_obstacle_penetration=self.evaluate_agent_obstacle_nonpenetration_constraint(
                sol, self.obstacles
            ),
            speed=self.evaluate_speed_constraint(sol),
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
                        self.obstacles.positions[obs_index, 0],
                        self.obstacles.positions[obs_index, 1],
                    ),
                    self.obstacles.radii[obs_index],
                    color="r",
                    alpha=0.5,
                )
            )

        # Plot the agents' trajectories
        for agent_index in range(self.num_agents):
            ax.plot(
                sol.agent_positions[:, agent_index, 0],
                sol.agent_positions[:, agent_index, 1],
                marker="o",
                label=f"Agent {agent_index}",
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

        ax.set_axis("equal")


def extract_solution_from_model(model) -> SolutionValue[np.ndarray]:
    return SolutionValue(
        agent_positions=np.array(
            [
                [
                    [value(model.AgentPositions[t, a, d]) for d in range(2)]
                    for a in range(model.Agent.last() + 1)
                ]
                for t in range(model.Time.last() + 1)
            ]
        ),
        agent_agent_distances=np.array(
            [
                {
                    (a1, a2): value(model.AgentAgentDistances[t, (a1, a2)])
                    for a1 in range(model.Agent.last() + 1)
                    for a2 in range(model.Agent.last() + 1)
                    if a2 > a1
                }
                for t in range(model.Time.last() + 1)
            ]
        ),
        agent_obstacle_distances=np.array(
            [
                [
                    [
                        value(model.AgentObjectDistances[t, a, o])
                        for o in range(model.Obstacle.last() + 1)
                    ]
                    for a in range(model.Agent.last() + 1)
                ]
                for t in range(model.Time.last() + 1)
            ]
        ),
    )


def solve_alm_subproblem(problem: Problem) -> SolutionResult:
    model: Any = ConcreteModel()

    # Define sets
    agent_collision_indices = [
        (time_index, source_agent_index, target_agent_index)
        for time_index in range(problem.num_timesteps - 1)
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
    model.AgentObjectDistances = Var(
        model.Time,
        model.Agent,
        model.Obstacle,
        within=NonNegativeReals,
    )
    model.AgentAgentDistances = Var(
        model.Time,
        model.AgentPairs,
        within=NonNegativeReals,
    )
    # Start and end position constraints
    model.StartPositionConstraint = Constraint(
        model.Agent,
        model.Dim,
        rule=lambda model, agent_index, dim: model.AgentPositions[0, agent_index, dim]
        == problem.agent_start_positions[agent_index, dim],
    )
    model.EndPositionConstraint = Constraint(
        model.Agent,
        model.Dim,
        rule=lambda model, agent_index, dim: model.AgentPositions[
            problem.num_timesteps - 1, agent_index, dim
        ]
        == problem.agent_end_positions[agent_index, dim],
    )
    symbolic_solution_value = SolutionValue.get_symbolic_from_model(model)
    symbolic_constraints = problem.get_constraints(symbolic_solution_value)
    model.SpeedConstraint = symbolic_constraints.get_speed_constraint(model)
    model.objective = Objective(
        rule=lambda _model: problem.evaluate_distance_objective(
            symbolic_solution_value
        ),
        sense=minimize,
    )

    # Solve the model
    solver: Ipopt = SolverFactory("ipopt")
    results = solver.solve(model, tee=False)

    return SolutionResult(
        value=SolutionValue.get_value_from_model(model),
        status=results.solver.status,
        termination_condition=results.solver.termination_condition,
    )


def run_alm(problem: Problem, rho: float, rho_factor: float, num_alm_iterations: int):
    """
    x: tensor of shape (batch_size, horizon, state_dim)
    """
    grad_o = []
    grad_a = []

    t_begin = time.time()
    success = 0
    for steps in range(num_alm_iterations):
        tt1 = time.time()

        # select only the agent index for what we're solving
        agent_idx = next(iter(init_traj4proj.keys()))
        agents_starts_pos_ = agents_starts_pos[agent_idx : agent_idx + 1, :]
        agents_goals_pos_ = agents_goals_pos[agent_idx : agent_idx + 1, :]
        agents_rads_ = agents_rads[agent_idx : agent_idx + 1]
        # agents_max_speeds = np.array([agents_max_speeds[agent_idx]])

        result = solve_alm_subproblem(
            x_candidate_alm,
            agents_starts_pos_,
            agents_goals_pos_,
            agents_rads_,
            agents_max_speeds,
            obs_pos,
            obs_rads,
            horizons,
            nu_AgentObstacleCollisions,
            nu_AgentAgentCollisions,
            rho,
            init_traj4proj_array,
            init_d_o,
            init_d_a,
            start_idx=start_idx,
            other_agents=other_agents,
            nu_dyn=nu_dyn,
            init_d_dyn=init_d_dyn,
        )

        # If solver failed, return original input without projection
        if solver_flag == 0:
            logger.warning(
                f"Solver failed at step {steps} - returning original input without projection"
            )
            return x, 0  # Return original x and success=0

        # Enforce prefix positions after optimization step
        if prefix_positions is not None and len(prefix_positions) > 1:
            # Assume planned agent is index 0
            # prefix_positions shape: (L, 2), where L <= horizons
            L = min(len(prefix_positions), x_iteration.shape[0])
            # Only overwrite the prefix part.
            # Note: x_iteration shape is (horizons, num_agents, 2)
            # Overwriting agent 0's path for the first L steps
            x_iteration[:L, 0, :] = prefix_positions[:L, :]

        grad_nu_o = d_o_dummy - d_o_real
        grad_nu_a = d_a_dummy - d_a_real
        grad_nu_dyn = d_dyn_dummy - d_dyn_real

        tolerance = proj_params["tolerance"]

        # threshold at zero: 1 if feasible (grad ≥ 0), else 0
        fea_o_real = np.sum((d_o_real <= -tolerance).astype(int))
        # We only care about violations after start_idx, so we should zero out any potential 'noise' before start_idx
        # (though loop above already ensures zeros before start_idx)
        fea_a_real = np.sum((d_a_real <= -tolerance).astype(int))
        fea_dyn_real = (
            np.sum((d_dyn_real <= -tolerance).astype(int))
            if len(other_agents) > 0
            else 0
        )

        init_traj4proj_array = x_iteration
        init_d_o, init_d_a, init_d_dyn = d_o_dummy, d_a_dummy, d_dyn_dummy

        grad_nu_o_norm = np.mean(np.linalg.norm(grad_nu_o))
        grad_nu_a_norm = np.mean(np.linalg.norm(grad_nu_a))
        grad_nu_dyn_norm = (
            np.mean(np.linalg.norm(grad_nu_dyn)) if num_dyn_obs > 0 else 0
        )
        logger.info(
            f"ALM step {steps}: {grad_nu_o_norm=}, {grad_nu_a_norm=}, {grad_nu_dyn_norm=}, {fea_o_real=}, {fea_a_real=}, {fea_dyn_real=}"
        )

        grad_nu_o_set.append(grad_nu_o_norm)
        grad_nu_a_set.append(grad_nu_a_norm)

        # Check convergence. Includes dynamic obstacles (norm_dyn, fea_dyn_real).
        grads_small = (
            grad_nu_o_norm <= 1e-3
            and grad_nu_a_norm <= 1e-3
            and grad_nu_dyn_norm <= 1e-3
        )
        constraints_ok = fea_o_real + fea_a_real + fea_dyn_real < 1
        if grads_small or constraints_ok:
            success = 1
            break

        # update the lagrange parameters
        rho = rho * rho_factor
        nu_AgentObstacleCollisions = nu_AgentObstacleCollisions + rho * grad_nu_o
        nu_AgentAgentCollisions = nu_AgentAgentCollisions + rho * grad_nu_a
        nu_dyn = nu_dyn + rho * grad_nu_dyn if num_dyn_obs > 0 else nu_dyn

        grad_o.append(grad_nu_o_norm)
        grad_a.append(grad_nu_a_norm)

        tt2 = time.time()

        logger.info(f"ALM step {steps} completed in {tt2 - tt1} seconds.")

    t_end = time.time()
    logger.info(f"Total time: {t_end - t_begin} seconds.")

    return x, success
