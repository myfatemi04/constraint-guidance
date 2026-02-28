import heapq
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ael.problem import Problem
from ael.solve import (
    DEFAULT_SCHEDULES,
    ScheduleEntry,
    ScoreComputationMethod,
    get_initial_paths_by_agent,
    solve,
)
from ael.visgraphprior import make_roadmap
from ael.visualize import visualize


@dataclass
class CBSConstraint:
    agent_i: int
    disallowed_location: np.ndarray
    disallowed_radius: float
    # time_step: int


@dataclass
class CBSNode:
    constraints: list[CBSConstraint]
    initial_paths: list[np.ndarray]


def make_constrained_graph(constraints: list[CBSConstraint], graph: nx.Graph):
    g = graph.copy()
    for n in g.nodes():
        if any(
            [
                np.linalg.norm(g.nodes[n]["pos"] - constraint.disallowed_location)
                < constraint.disallowed_radius
                for constraint in constraints
            ]
        ):
            g.remove_node(n)
    return g


def cbs_spatial_approximation(
    problem: Problem,
    score_computation_method: ScoreComputationMethod,
    schedule: list[ScheduleEntry],
):
    graph = make_roadmap(problem)
    initial_paths = get_initial_paths_by_agent(problem, dt=1.0, graph=graph)
    initial_cost = (
        np.linalg.norm(initial_paths[1:] - initial_paths[:-1], axis=-1)
        # Sum across timesteps.
        .sum(axis=-1)
        # Sum across agents.
        .sum(axis=-1)
    )

    nodes = [(initial_cost, CBSNode(constraints=[], initial_paths=initial_paths))]

    while len(nodes) > 0:
        cost, node = heapq.heappop(nodes)
        # Try to solve this node.
        result = solve(
            problem,
            score_computation_method,
            schedule=schedule,
            initial_paths=node.initial_paths,
        )
        # Check for constraint violations. (t, a, a)
        aa_constraint_residuals_T_A_A = (
            result.constraint_satisfaction.agent_agent_constraint_residuals
        )
        # Find the earliest constraint violation.
        violation_indices = np.argwhere(aa_constraint_residuals_T_A_A > 1e-4)
        print("visualizing")
        # Visualize result.
        visualize(problem, plt.gca(), agent_positions=result.trajectories[-1])
        traj = result.trajectories[-1]
        # mark the first violation index
        if violation_indices.shape[0] > 0:
            t, a1, a2 = violation_indices[0]
            plt.plot(traj[t, [a1, a2], 0], traj[t, [a1, a2], 1], "ro", markersize=10)
        plt.show()

        if violation_indices.shape[0] == 0:
            # No constraint violations, return this solution.
            return result
        else:
            t, a1_, a2_ = violation_indices[0]

            def add_cbs_subnodes(a1, a2):
                # Create a new constraint for a1 and add it to the list of constraints.
                r = problem.agent_radii[a1] + problem.agent_radii[a2]
                new_constraint_a1 = CBSConstraint(
                    agent_i=a1, disallowed_location=traj[a2, t], disallowed_radius=r
                )
                new_constraints_a1 = node.constraints + [new_constraint_a1]
                constrained_graph_a1 = make_constrained_graph(new_constraints_a1, graph)
                constrained_paths_a1 = get_initial_paths_by_agent(
                    problem, dt=1.0, graph=constrained_graph_a1
                )
                cost_a1 = (
                    np.linalg.norm(
                        constrained_paths_a1[1:] - constrained_paths_a1[:-1], axis=-1
                    )
                    # Sum across timesteps.
                    .sum(axis=-1)
                    # Sum across agents.
                    .sum(axis=-1)
                )
                node_a1 = CBSNode(
                    constraints=new_constraints_a1, initial_paths=constrained_paths_a1
                )
                heapq.heappush(nodes, (cost_a1, node_a1))

            add_cbs_subnodes(a1_, a2_)
            add_cbs_subnodes(a2_, a1_)


if __name__ == "__main__":
    with open("instances_data/instances_dense.json", "r") as f:
        data = json.load(f)
    # a problem that was found not to converge originally
    hard_problem = Problem.from_json(data[284])
    result = cbs_spatial_approximation(
        hard_problem,
        score_computation_method=ScoreComputationMethod.APPROXIMATE_V0,
        schedule=DEFAULT_SCHEDULES[ScoreComputationMethod.APPROXIMATE_V0],
    )
