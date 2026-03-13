from typing import TYPE_CHECKING

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,  # type: ignore
    Objective,
    Param,
    RangeSet,
    Set,
    SolverFactory,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

if TYPE_CHECKING:
    from ael.optimize import Problem


def solve_L_aug(
    x,
    agent_starts,
    agent_goals,
    agent_rads,
    max_speeds,
    obs_pos,
    obs_rads,
    horizons,
    nu_o,
    nu_a,
    rho,
    init_traj4proj_array,
    init_d_o,
    init_d_a,
    fixed_prefix_positions=None,
    fixed_suffix_positions=None,
    start_idx=0,
    end_idx=None,
    other_agents=None,
    nu_dyn=None,
    init_d_dyn=None,
):
    num_agents = agent_starts.shape[0]
    num_obs = obs_pos.shape[0]
    max_speeds = max_speeds * np.ones(num_agents)

    # Planning window (inclusive start, exclusive end)
    horizons = int(horizons)
    start_idx = int(start_idx)
    if end_idx is None:
        end_idx = horizons
    end_idx = int(end_idx)
    start_idx = max(0, min(start_idx, horizons))
    end_idx = max(0, min(end_idx, horizons))
    if end_idx < start_idx:
        end_idx = start_idx

    # Dynamic obstacles handling
    if other_agents is None:
        other_agents = []
    num_dyn_obs = len(other_agents)

    model = ConcreteModel()

    # Define sets
    model.I = RangeSet(0, horizons - 1)  # Time steps
    model.J = RangeSet(0, num_agents - 1)  # Agents
    model.K = RangeSet(0, num_obs - 1)  # Obstacles
    model.Dim = RangeSet(0, 1)  # Dimensions (x and y)
    model.AgentPairs = Set(
        initialize=[
            (j, k) for j in range(num_agents) for k in range(num_agents) if k > j
        ]
    )
    model.KDyn = RangeSet(0, max(0, num_dyn_obs - 1))  # Dynamic Obstacles indices

    initial_p_out = {
        (i, j, k): init_traj4proj_array[i, j, k]
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(2)
    }
    initial_d_o = {
        (i, j, k): init_d_o[i, j, k] + 1e-8
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(num_obs)
    }
    initial_d_a = {
        (i, (j, k)): init_d_a[i, j, k] + 1e-8
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(num_agents)
        if k > j
    }

    # Initialize d_dyn
    initial_d_dyn = {}
    if num_dyn_obs > 0:
        for j in range(num_agents):
            for k in range(num_dyn_obs):
                initial_d_dyn[(j, k)] = init_d_dyn[j, k] + 1e-8

    model.P_out = Var(
        model.I, model.J, model.Dim, bounds=(-1, 1), initialize=initial_p_out
    )
    model.d_o = Var(
        model.I, model.J, model.K, within=NonNegativeReals, initialize=initial_d_o
    )
    model.d_a = Var(
        model.I, model.AgentPairs, within=NonNegativeReals, initialize=initial_d_a
    )

    if num_dyn_obs > 0:
        model.d_dyn = Var(
            model.J, model.KDyn, within=NonNegativeReals, initialize=initial_d_dyn
        )
    else:
        # Create dummy variable to avoid Pyomo errors if accessed in loops (though we guard loops)
        model.d_dyn = Var(model.J, model.KDyn, within=NonNegativeReals)

    # Fix prefix/suffix positions for the planned agent (assumed agent index 0).
    # This ensures replan does not modify points outside the active window.
    if num_agents > 0:
        if fixed_prefix_positions is not None:
            try:
                prefix_arr = np.asarray(fixed_prefix_positions, dtype=float)
                if prefix_arr.ndim == 2 and prefix_arr.shape[0] > 0:
                    L = min(int(prefix_arr.shape[0]), horizons)
                    for i in range(L):
                        model.P_out[i, 0, 0].fix(float(prefix_arr[i, 0]))
                        model.P_out[i, 0, 1].fix(float(prefix_arr[i, 1]))
            except Exception:
                pass
        if fixed_suffix_positions is not None:
            try:
                suffix_arr = np.asarray(fixed_suffix_positions, dtype=float)
                if suffix_arr.ndim == 2 and suffix_arr.shape[0] > 0:
                    S = min(int(suffix_arr.shape[0]), horizons)
                    start_t = horizons - S
                    for idx in range(S):
                        t = start_t + idx
                        model.P_out[t, 0, 0].fix(float(suffix_arr[idx, 0]))
                        model.P_out[t, 0, 1].fix(float(suffix_arr[idx, 1]))
            except Exception:
                pass

    # Parameters (converted from arrays to dictionaries for Pyomo)
    agent_rads_dict = {j: agent_rads[j] for j in range(num_agents)}
    model.agent_rads = Param(model.J, initialize=agent_rads_dict)

    max_speeds_dict = {j: max_speeds[j] for j in range(num_agents)}
    model.max_speeds = Param(model.J, initialize=max_speeds_dict)

    obs_rads_dict = {k: obs_rads[k] for k in range(num_obs)}
    model.obs_rads = Param(model.K, initialize=obs_rads_dict)

    obs_pos_dict_x = {k: obs_pos[k, 0] for k in range(num_obs)}
    obs_pos_dict_y = {k: obs_pos[k, 1] for k in range(num_obs)}
    model.obs_pos_x = Param(model.K, initialize=obs_pos_dict_x)
    model.obs_pos_y = Param(model.K, initialize=obs_pos_dict_y)

    # Dynamic obstacles parameters
    dyn_obs_x_dict = {}
    dyn_obs_y_dict = {}
    dyn_obs_rad_dict = {}
    dyn_obs_t_dict = {}
    for k, obs in enumerate(other_agents):
        # obs: [pos, rad, time_step]
        dyn_obs_x_dict[k] = obs[0][0]
        dyn_obs_y_dict[k] = obs[0][1]
        dyn_obs_rad_dict[k] = float(obs[1])
        dyn_obs_t_dict[k] = int(obs[2])

    model.dyn_obs_x = Param(model.KDyn, initialize=dyn_obs_x_dict)
    model.dyn_obs_y = Param(model.KDyn, initialize=dyn_obs_y_dict)
    model.dyn_obs_rad = Param(model.KDyn, initialize=dyn_obs_rad_dict)
    model.dyn_obs_t = Param(model.KDyn, initialize=dyn_obs_t_dict)

    # Lagrangian multipliers
    nu_o_dict = {
        (i, j, k): nu_o[i, j, k]
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(num_obs)
    }
    model.nu_o = Param(model.I, model.J, model.K, initialize=nu_o_dict)

    nu_a_dict = {
        (i, j, k): nu_a[i, j, k]
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(num_agents)
        if k > j
    }
    model.nu_a = Param(model.I, model.AgentPairs, initialize=nu_a_dict)

    nu_dyn_dict = {}
    if num_dyn_obs > 0:
        for j in range(num_agents):
            for k in range(num_dyn_obs):
                nu_dyn_dict[(j, k)] = nu_dyn[j, k]
    model.nu_dyn = Param(model.J, model.KDyn, initialize=nu_dyn_dict)

    # Objective function
    def obj_expression(model):
        expr = 0.0
        # Minimize the distance between predicted path and the true path
        for i in model.I:
            if i < start_idx or i >= end_idx:
                continue  # Skip fixed prefix/suffix in cost
            for j in model.J:
                expr += sum((model.P_out[i, j, k] - x[i, j, k]) ** 2 for k in model.Dim)
        # minimize total distance
        for i in model.I:
            if i < start_idx or i >= end_idx:
                continue  # Skip fixed prefix/suffix in cost
            for j in model.J:
                if i < model.I.last():
                    expr += sum(
                        (model.P_out[i + 1, j, k] - model.P_out[i, j, k]) ** 2
                        for k in model.Dim
                    )
        # Lagrangian terms for obstacle avoidance
        for i in model.I:
            if i < start_idx or i >= end_idx:
                continue  # Skip fixed prefix/suffix in cost
            for j in model.J:
                for k in model.K:
                    temp_expr = (
                        -(
                            (model.P_out[i, j, 0] - model.obs_pos_x[k]) ** 2
                            + (model.P_out[i, j, 1] - model.obs_pos_y[k]) ** 2
                        )
                        + (model.agent_rads[j] + model.obs_rads[k]) ** 2
                        + model.d_o[i, j, k]
                    )
                    expr += model.nu_o[i, j, k] * temp_expr
        # Lagrangian terms for collision avoidance
        for i in model.I:
            if i < start_idx or i >= end_idx:
                continue  # Skip fixed prefix/suffix in cost
            for j, k in model.AgentPairs:
                temp_expr = (
                    -(
                        (model.P_out[i, j, 0] - model.P_out[i, k, 0]) ** 2
                        + (model.P_out[i, j, 1] - model.P_out[i, k, 1]) ** 2
                    )
                    + (model.agent_rads[j] + model.agent_rads[k]) ** 2
                    + model.d_a[i, (j, k)]
                )
                expr += model.nu_a[i, (j, k)] * temp_expr

        # Lagrangian terms for dynamic obstacles
        if num_dyn_obs > 0:
            for j in model.J:
                for k in model.KDyn:
                    t = model.dyn_obs_t[k]
                    # Only consider if t is within planning window
                    if start_idx <= t < end_idx:
                        temp_expr = (
                            -(
                                (model.P_out[t, j, 0] - model.dyn_obs_x[k]) ** 2
                                + (model.P_out[t, j, 1] - model.dyn_obs_y[k]) ** 2
                            )
                            + (model.agent_rads[j] + model.dyn_obs_rad[k]) ** 2
                            + model.d_dyn[j, k]
                        )
                        expr += model.nu_dyn[j, k] * temp_expr

        # Augmented Lagrangian terms for obstacle avoidance
        for i in model.I:
            if i < start_idx or i >= end_idx:
                continue  # Skip fixed prefix/suffix in cost
            for j in model.J:
                for k in model.K:
                    temp_expr = (
                        -(
                            (model.P_out[i, j, 0] - model.obs_pos_x[k]) ** 2
                            + (model.P_out[i, j, 1] - model.obs_pos_y[k]) ** 2
                        )
                        + (model.agent_rads[j] + model.obs_rads[k]) ** 2
                        + model.d_o[i, j, k]
                    )
                    expr += rho * temp_expr**2
        # Augmented Lagrangian terms for collision avoidance
        for i in model.I:
            if i < start_idx or i >= end_idx:
                continue  # Skip fixed prefix/suffix in cost
            for j, k in model.AgentPairs:
                temp_expr = (
                    -(
                        (model.P_out[i, j, 0] - model.P_out[i, k, 0]) ** 2
                        + (model.P_out[i, j, 1] - model.P_out[i, k, 1]) ** 2
                    )
                    + (model.agent_rads[j] + model.agent_rads[k]) ** 2
                    + model.d_a[i, (j, k)]
                )
                expr += rho * temp_expr**2

        # Augmented Lagrangian terms for dynamic obstacles
        if num_dyn_obs > 0:
            for j in model.J:
                for k in model.KDyn:
                    t = model.dyn_obs_t[k]
                    if start_idx <= t < end_idx:
                        temp_expr = (
                            -(
                                (model.P_out[t, j, 0] - model.dyn_obs_x[k]) ** 2
                                + (model.P_out[t, j, 1] - model.dyn_obs_y[k]) ** 2
                            )
                            + (model.agent_rads[j] + model.dyn_obs_rad[k]) ** 2
                            + model.d_dyn[j, k]
                        )
                        expr += rho * temp_expr**2

        return expr

    model.obj = Objective(rule=obj_expression, sense=minimize)

    # Start and end position constraints
    agent_starts_x = {j: agent_starts[j, 0] for j in range(num_agents)}
    agent_starts_y = {j: agent_starts[j, 1] for j in range(num_agents)}
    agent_goals_x = {j: agent_goals[j, 0] for j in range(num_agents)}
    agent_goals_y = {j: agent_goals[j, 1] for j in range(num_agents)}

    def start_position_constraint_x(model, j):
        return model.P_out[0, j, 0] == agent_starts_x[j]

    def start_position_constraint_y(model, j):
        return model.P_out[0, j, 1] == agent_starts_y[j]

    def end_position_constraint_x(model, j):
        return model.P_out[horizons - 1, j, 0] == agent_goals_x[j]

    def end_position_constraint_y(model, j):
        return model.P_out[horizons - 1, j, 1] == agent_goals_y[j]

    model.start_position_constraints_x = Constraint(
        model.J, rule=start_position_constraint_x
    )
    model.start_position_constraints_y = Constraint(
        model.J, rule=start_position_constraint_y
    )
    model.end_position_constraints_x = Constraint(
        model.J, rule=end_position_constraint_x
    )
    model.end_position_constraints_y = Constraint(
        model.J, rule=end_position_constraint_y
    )

    # Note: prefix/suffix fixing is done via Var.fix() above.

    # Speed constraints
    def speed_constraint_rule(model, i, j):
        if i < horizons - 1:
            return (
                sum(
                    (model.P_out[i, j, k] - model.P_out[i + 1, j, k]) ** 2
                    for k in model.Dim
                )
                <= model.max_speeds[j] ** 2
            )
        else:
            return Constraint.Skip

    model.speed_constraints = Constraint(model.I, model.J, rule=speed_constraint_rule)

    # Solve the model
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "mumps"
    results = solver.solve(model, tee=False)

    # Extract solution
    x_iteration = np.zeros((horizons, num_agents, 2))
    d_o_value = np.zeros((horizons, num_agents, num_obs))
    d_a_value = np.zeros((horizons, num_agents, num_agents))
    d_dyn_value = np.zeros((num_agents, max(1, num_dyn_obs)))  # Ensure at least size 1

    flag = 0

    if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal
    ):
        flag = 1
        for i in range(horizons):
            for j in range(num_agents):
                x_iteration[i, j, 0] = value(model.P_out[i, j, 0])
                x_iteration[i, j, 1] = value(model.P_out[i, j, 1])
                for k_obs in range(num_obs):
                    d_o_value[i, j, k_obs] = value(model.d_o[i, j, k_obs])
                for k_agent in range(num_agents):
                    if k_agent > j:
                        d_a_value[i, j, k_agent] = value(model.d_a[i, (j, k_agent)])

        if num_dyn_obs > 0:
            for j in range(num_agents):
                for k in range(num_dyn_obs):
                    d_dyn_value[j, k] = value(model.d_dyn[j, k])
    else:
        # Solver failed
        flag = 0
        return x_iteration, d_o_value, d_a_value, d_dyn_value, flag
    return x_iteration, d_o_value, d_a_value, d_dyn_value, flag


def grad_nu(
    agent_rads,
    obs_pos,
    obs_rads,
    p_value,
    d_o,
    d_a,
    start_idx=0,
    end_idx=None,
    other_agents=None,
    d_dyn=None,
):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0]
    start_idx = int(start_idx)
    if end_idx is None:
        end_idx = horizons
    end_idx = int(end_idx)
    start_idx = max(0, min(start_idx, horizons))
    end_idx = max(0, min(end_idx, horizons))
    if end_idx < start_idx:
        end_idx = start_idx

    # initialize the gradient of nu_o and nu_a
    grad_nu_o = np.zeros((horizons, num_agents, num_obs))
    grad_nu_a = np.zeros((horizons, num_agents, num_agents))

    if other_agents is None:
        other_agents = []
    num_dyn = len(other_agents)
    grad_nu_dyn = np.zeros((num_agents, max(1, num_dyn)))

    # calculate the gradient of nu_o and nu_a
    for i in range(start_idx, end_idx):
        for j in range(num_agents):
            for k in range(num_obs):
                grad_nu_o[i, j, k] = (
                    -(
                        (p_value[i, j, 0] - obs_pos[k, 0]) ** 2
                        + (p_value[i, j, 1] - obs_pos[k, 1]) ** 2
                    )
                    + (agent_rads[j] + obs_rads[k]) ** 2
                    + d_o[i, j, k]
                )
            for k in range(j + 1, num_agents):
                grad_nu_a[i, j, k] = (
                    -(
                        (p_value[i, j, 0] - p_value[i, k, 0]) ** 2
                        + (p_value[i, j, 1] - p_value[i, k, 1]) ** 2
                    )
                    + (agent_rads[j] + agent_rads[k]) ** 2
                    + d_a[i, j, k]
                )

    if num_dyn > 0:
        for j in range(num_agents):
            for k in range(num_dyn):
                obs = other_agents[k]
                pos = obs[0]
                rad = obs[1]
                t = int(obs[2])
                if start_idx <= t < end_idx:
                    grad_nu_dyn[j, k] = (
                        -(
                            (p_value[t, j, 0] - pos[0]) ** 2
                            + (p_value[t, j, 1] - pos[1]) ** 2
                        )
                        + (agent_rads[j] + rad) ** 2
                        + d_dyn[j, k]
                    )

    return grad_nu_o, grad_nu_a, grad_nu_dyn


def check_feasibility(
    agent_rads,
    obs_pos,
    obs_rads,
    p_value,
    d_o,
    d_a,
    tolerance,
    start_idx=0,
    end_idx=None,
    other_agents=None,
):
    horizons = p_value.shape[0]
    num_agents = p_value.shape[1]
    num_obs = obs_pos.shape[0]
    start_idx = int(start_idx)
    if end_idx is None:
        end_idx = horizons
    end_idx = int(end_idx)
    start_idx = max(0, min(start_idx, horizons))
    end_idx = max(0, min(end_idx, horizons))
    if end_idx < start_idx:
        end_idx = start_idx

    if other_agents is None:
        other_agents = []
    num_dyn = len(other_agents)

    #

    # initialize the gradient of nu_o and nu_a
    grad_nu_o = np.zeros((horizons, num_agents, num_obs))
    grad_nu_a = np.zeros((horizons, num_agents, num_agents))
    grad_nu_dyn = np.zeros((num_agents, max(1, num_dyn)))

    # calculate the gradient of nu_o and nu_a
    for i in range(start_idx, end_idx):
        for j in range(num_agents):
            for k in range(num_obs):
                grad_nu_o[i, j, k] = (
                    (p_value[i, j, 0] - obs_pos[k, 0]) ** 2
                    + (p_value[i, j, 1] - obs_pos[k, 1]) ** 2
                ) - (agent_rads[j] + obs_rads[k]) ** 2
            for k in range(j + 1, num_agents):
                grad_nu_a[i, j, k] = (
                    (p_value[i, j, 0] - p_value[i, k, 0]) ** 2
                    + (p_value[i, j, 1] - p_value[i, k, 1]) ** 2
                ) - (agent_rads[j] + agent_rads[k]) ** 2

    if num_dyn > 0:
        for j in range(num_agents):
            for k in range(num_dyn):
                obs = other_agents[k]
                pos = obs[0]
                rad = obs[1]
                t = int(obs[2])
                if start_idx <= t < end_idx:
                    grad_nu_dyn[j, k] = (
                        (p_value[t, j, 0] - pos[0]) ** 2
                        + (p_value[t, j, 1] - pos[1]) ** 2
                    ) - (agent_rads[j] + rad) ** 2

    # threshold at zero: 1 if feasible (grad ≥ 0), else 0
    feas_o = (grad_nu_o <= -tolerance).astype(int)
    # We only care about violations after start_idx, so we should zero out any potential 'noise' before start_idx
    # (though loop above already ensures zeros before start_idx)
    feas_a = (grad_nu_a <= -tolerance).astype(int)

    feas_dyn = 0
    if num_dyn > 0:
        feas_dyn = np.sum((grad_nu_dyn <= -tolerance).astype(int))

    # Sum only considers indices where we calculated values
    return np.sum(feas_o), np.sum(feas_a), feas_dyn


def cal_dummy_var(
    agent_rads,
    obs_pos,
    obs_rads,
    agent_pos,
    start_idx=0,
    end_idx=None,
    other_agents=None,
):
    horizons = agent_pos.shape[0]
    num_agents = agent_pos.shape[1]
    num_obs = obs_pos.shape[0]
    start_idx = int(start_idx)
    if end_idx is None:
        end_idx = horizons
    end_idx = int(end_idx)
    start_idx = max(0, min(start_idx, horizons))
    end_idx = max(0, min(end_idx, horizons))
    if end_idx < start_idx:
        end_idx = start_idx

    if other_agents is None:
        other_agents = []
    num_dyn = len(other_agents)

    #

    # initialize the gradient of nu_o and nu_a
    d_o = np.zeros((horizons, num_agents, num_obs))
    d_a = np.zeros((horizons, num_agents, num_agents))
    d_dyn = np.zeros((num_agents, max(1, num_dyn)))

    # calculate the gradient of nu_o and nu_a
    for i in range(start_idx, end_idx):
        for j in range(num_agents):
            for k in range(num_obs):
                d_o[i, j, k] = (
                    (agent_pos[i, j, 0] - obs_pos[k, 0]) ** 2
                    + (agent_pos[i, j, 1] - obs_pos[k, 1]) ** 2
                ) - (agent_rads[j] + obs_rads[k]) ** 2
            for k in range(j + 1, num_agents):
                d_a[i, j, k] = (
                    (agent_pos[i, j, 0] - agent_pos[i, k, 0]) ** 2
                    + (agent_pos[i, j, 1] - agent_pos[i, k, 1]) ** 2
                ) - (agent_rads[j] + agent_rads[k]) ** 2

    if num_dyn > 0:
        for j in range(num_agents):
            for k in range(num_dyn):
                obs = other_agents[k]
                pos = obs[0]
                rad = obs[1]
                t = int(obs[2])
                if start_idx <= t < end_idx:
                    d_dyn[j, k] = (
                        (agent_pos[t, j, 0] - pos[0]) ** 2
                        + (agent_pos[t, j, 1] - pos[1]) ** 2
                    ) - (agent_rads[j] + rad) ** 2

    return d_o, d_a, d_dyn


def apply_projection_alm(problem: "Problem"):
    # rebuttal
    grad_nu_o_set = []
    grad_nu_a_set = []

    num_timesteps = problem.num_timesteps
    num_agents = problem.num_agents
    num_obs = problem.num_circular_obstacles
    agents_starts_pos = problem.agent_start_positions
    agents_goals_pos = problem.agent_end_positions

    # Optional fixed prefix/suffix from proj_params (assumed for planned agent index 0)
    prefix_positions = None
    suffix_positions = None
    start_idx = 0
    end_idx = num_timesteps

    start_idx = int(max(0, min(start_idx, num_timesteps)))
    end_idx = int(max(0, min(end_idx, num_timesteps)))
    if end_idx < start_idx:
        end_idx = start_idx

    # Initialize the lagrange multipliers
    nu_o = np.zeros((num_timesteps, num_agents, num_obs))
    nu_a = np.zeros((num_timesteps, num_agents, num_agents))
    p_init = np.zeros((num_timesteps, num_agents, 2))
    d_o = np.zeros((num_timesteps, num_agents, num_obs))
    d_a = np.zeros((num_timesteps, num_agents, num_agents))

    for i in range(num_timesteps):
        for j in range(num_agents):
            p_init[i, j, :] = agents_starts_pos[j, :] + (
                agents_goals_pos[j, :] - agents_starts_pos[j, :]
            ) * (i / (num_timesteps - 1))

    init_d_o, init_d_a, init_d_dyn = cal_dummy_var(
        problem.agent_radii,
        problem.circular_obstacle_positions,
        problem.circular_obstacle_radii,
        problem.agent_reference_trajectory,
        start_idx=start_idx,
        end_idx=end_idx,
        other_agents=None,
    )
    init_d_o = np.maximum(init_d_o, 0)
    init_d_a = np.maximum(init_d_a, 0)
    if init_d_dyn is not None:
        init_d_dyn = np.maximum(init_d_dyn, 0)

    # Set lagrange parameters
    # rho = proj_params["rho"]
    # rho_factor = proj_params["rho_factor"]
    rho = 5.0
    rho_factor = 1.05
    alm_iteration = 10
    tolerance = 1e-5

    agents_rads = problem.agent_radii
    agents_max_speeds = problem.agent_max_speeds
    obs_pos = problem.circular_obstacle_positions
    obs_rads = problem.circular_obstacle_radii
    x_iteration = problem.agent_reference_trajectory.copy()
    other_agents = None
    nu_dyn = None

    grad_o = []
    grad_a = []
    import time

    t_begin = time.time()
    success = 0
    for steps in range(alm_iteration):
        tt1 = time.time()

        x_iteration, d_o, d_a, d_dyn, solver_flag = solve_L_aug(
            x_iteration,
            agents_starts_pos,
            agents_goals_pos,
            agents_rads,
            agents_max_speeds,
            obs_pos,
            obs_rads,
            num_timesteps,
            nu_o,
            nu_a,
            rho,
            x_iteration,
            d_o,
            d_a,
            fixed_prefix_positions=prefix_positions,
            fixed_suffix_positions=suffix_positions,
            start_idx=start_idx,
            end_idx=end_idx,
            other_agents=other_agents,
            nu_dyn=nu_dyn,
            init_d_dyn=init_d_dyn,
        )

        # If solver failed, return original input without projection
        if solver_flag == 0:
            print(
                "WARNING: Solver failed at step",
                steps,
                "- returning original input without projection",
            )
            return (
                problem.agent_reference_trajectory,
                0,
            )  # Return original x and success=0

        # Enforce prefix/suffix positions after optimization step (safety; primary fix is in solve_L_aug via Var.fix()).
        if prefix_positions is not None and len(prefix_positions) > 0:
            # Assume planned agent is index 0
            # prefix_positions shape: (L, 2), where L <= horizons
            L = min(len(prefix_positions), x_iteration.shape[0])
            # Only overwrite the prefix part.
            # Note: x_iteration shape is (horizons, num_agents, 2)
            # Overwriting agent 0's path for the first L steps
            x_iteration[:L, 0, :] = prefix_positions[:L, :]
        if suffix_positions is not None and len(suffix_positions) > 0:
            S = min(len(suffix_positions), x_iteration.shape[0])
            start_t = x_iteration.shape[0] - S
            x_iteration[start_t:, 0, :] = suffix_positions[:S, :]

        grad_nu_o, grad_nu_a, grad_nu_dyn = grad_nu(
            problem.agent_radii,
            problem.circular_obstacle_positions,
            problem.circular_obstacle_radii,
            x_iteration,
            d_o,
            d_a,
            start_idx=start_idx,
            end_idx=end_idx,
            other_agents=other_agents,
            d_dyn=d_dyn,
        )

        init_d_o, init_d_a, init_d_dyn = d_o, d_a, d_dyn

        fea_o_real, fea_a_real, fea_dyn_real = check_feasibility(
            agents_rads,
            obs_pos,
            problem.circular_obstacle_radii,
            x_iteration,
            d_o,
            d_a,
            tolerance,
            start_idx=start_idx,
            end_idx=end_idx,
            other_agents=other_agents,
        )

        # # norm
        print(steps)
        print("grad for solving alm")
        print(np.mean(np.linalg.norm(grad_nu_o)))
        print(np.mean(np.linalg.norm(grad_nu_a)))
        print(np.mean(np.linalg.norm(grad_nu_dyn)))
        print("grad for real settings")
        print(np.mean(fea_o_real))
        print(np.mean(fea_a_real))
        print(np.mean(fea_dyn_real))

        grad_nu_o_set.append(np.mean(np.linalg.norm(grad_nu_o)))
        grad_nu_a_set.append(np.mean(np.linalg.norm(grad_nu_a)))

        # norm_dyn = 0
        # if num_dyn_obs > 0:
        #     norm_dyn = np.mean(np.linalg.norm(grad_nu_dyn))

        # Check convergence: gradients are small OR constraints are satisfied
        # Includes dynamic obstacles (norm_dyn, fea_dyn_real)
        if (
            np.mean(np.linalg.norm(grad_nu_o)) <= 1e-3
            and np.mean(np.linalg.norm(grad_nu_a)) <= 1e-3
            # and norm_dyn <= 1e-3
        ) or (fea_o_real + fea_a_real + fea_dyn_real < 1):
            success = 1
            break

        else:
            # update the lagrange parameters
            rho = rho * rho_factor
            nu_o = nu_o + rho * grad_nu_o
            nu_a = nu_a + rho * grad_nu_a
            # if num_dyn_obs > 0:
            #     nu_dyn = nu_dyn + rho * grad_nu_dyn

            grad_o.append(np.mean(np.linalg.norm(grad_nu_o)))
            grad_a.append(np.mean(np.linalg.norm(grad_nu_a)))

        tt2 = time.time()
        print("Time for each iteration:", tt2 - tt1)

    t_end = time.time()
    print("Time:", t_end - t_begin)

    return x_iteration, success
