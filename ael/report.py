"""For generating scientific figures."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_tolerances(num_robots, tolerance_levels, df, title):
    plt.title(f"{num_robots} Robots, {title}")

    for i_constraint_type, constraint_type in enumerate(
        ["agent_obstacle", "agent_agent", "velocity"]
    ):
        tolerance_results = []

        for tol in tolerance_levels:
            subset = df[df["num_robots"] == num_robots]
            satisfied = subset[f"{constraint_type}_max_residual"] < tol
            fraction_satisfied = satisfied.sum() / len(subset)
            tolerance_results.append(fraction_satisfied)

        plt.plot(tolerance_levels, tolerance_results, marker="o", label=constraint_type)

    plt.xscale("log")
    plt.yscale("linear")
    plt.xlabel("Constraint Satisfaction Tolerance")
    plt.ylabel("Fraction of Problems Satisfied")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()


def generate_constraint_satisfaction_figure(
    df: pd.DataFrame, title: str, tolerance_levels=[1e-2, 1e-3, 1e-4, 1e-5]
):
    num_agents_list = sorted(df["num_robots"].unique())

    plt.figure(figsize=(4 * len(num_agents_list), 4))

    # 3 rows, 3 columns: row == num_robots, col == constraint type.
    for i_num_robots, num_robots in enumerate(num_agents_list):
        plt.subplot(1, len(num_agents_list), 1 + i_num_robots)
        plot_tolerances(num_robots, tolerance_levels, df, title)

    plt.tight_layout()
    # plt.show()


def generate_simple_latex_table(
    data: dict, columns: list[tuple[str, str]] | None = None
) -> str:
    """
    Generates a simple LaTeX table.
    Args:
        data: A dictionary mapping column names to lists of values.
        columns: A list of tuples, where each tuple contains the column name and its LaTeX representation. This determines the order of columns in the table. If not provided, columns in `data` are used in Python iteration order.
    """

    if columns is None:
        columns = [(col, col) for col in data.keys()]

    num_rows = len(data[columns[0][0]])
    table: str = r"\begin{tabular}{|" + "c|" * len(columns) + "}\n\\hline\n"
    table += " & ".join([col[1] for col in columns]) + r" \\" + "\n\\hline\n"

    for i in range(num_rows):
        row_values = [str(data[col[0]][i]) for col in columns]
        table += " & ".join(row_values) + r" \\" + "\n"

    table += r"\hline" + "\n" + r"\end{tabular}"
    return table


def make_paths(base):
    return {
        k: f"results/{base}/{k}/table.csv"
        for k in os.listdir(f"results/{base}")
        if os.path.isdir(f"results/{base}/{k}")
    }


def main():
    path_groups = {
        # without Voronoi initialization
        "000_no_voronoi_initialization": {
            "dense": "results/2026-01-29/experiment_16-18-37_dense_num_robots=any/table.csv",
            "connected_room": "results/2026-01-30/experiment_13-32-20_connected_room_num_robots=any/table.csv",
            "shelf": "results/2026-01-30/experiment_13-27-10_shelf_num_robots=any/table.csv",
            "simple": "results/2026-01-29/experiment_16-44-37_simple_num_robots=any/table.csv",
        },
        # with Voronoi initialization
        "001_voronoi_initialization": {
            "connected_room": "results/2026-02-24/experiment_07-22-43_connected_room_num_robots=any/table.csv",
            "dense": "results/2026-02-24/experiment_07-31-38_dense_num_robots=any/table.csv",
            "shelf": "results/2026-02-24/experiment_07-36-12_shelf_num_robots=any/table.csv",
            "simple": "results/2026-02-24/experiment_07-39-32_simple_num_robots=any/table.csv",
        },
        "002_better_schedule": {
            k: f"results/2026-02-24/experiment_08-04-06/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "003_better_schedule_2": {
            k: f"results/2026-02-24/experiment_08-26-02/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "004_velocity_constraint_wrong_clipping": {
            k: f"results/2026-02-26/experiment_19-42-48/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "005_velocity_constraint": {
            k: f"results/2026-02-27/experiment_19-04-11/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "006_factorized_mppi": {
            k: f"results/2026-02-28/experiment_07-31-02/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "007_more_factorized_mppi": {
            k: f"results/2026-02-28/experiment_08-25-45/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "008_reproduce_005": {
            k: f"results/2026-02-28/experiment_09-55-02/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "009_highlevel_search": {
            k: f"results/2026-02-28/experiment_11-22-03_APPROXIMATE_V0_cbs_spatial_approximation/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "010_highlevel_search_more": {
            k: f"results/2026-03-03/experiment_11-48-55_APPROXIMATE_V0_cbs_spatial_approximation/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "011_alm": {
            k: f"results/2026-03-12/experiment_21-50-02_alm/{k}/table.csv"
            for k in ["dense", "simple", "shelf", "connected_room"]
        },
        "012_voronoi_baseline": make_paths(
            "2026-03-12/experiment_21-58-10_NONE_BASELINE_none"
        ),
        "013_larger_maps": make_paths(
            "2026-03-14/experiment_14-20-47_APPROXIMATE_V0_none"
        ),
    }

    name = sorted(path_groups.keys())[-1]
    paths = path_groups[name]

    df_and_title = [
        (pd.read_csv(paths[key]), key.replace("_", " ").title())
        for key in paths.keys()
        if os.path.exists(paths[key])
    ]

    figures_dir = Path("figures") / name
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 4))
    for i, (df, title) in enumerate(df_and_title):
        num_robots = df["num_robots"].unique()[0]
        plt.subplot(1, 4, i + 1)
        plot_tolerances(num_robots, [1e-2, 1e-3, 1e-4, 1e-5], df, title)
        # generate_constraint_satisfaction_figure(df, title)
        # plt.savefig(
        #     figures_dir
        #     / f"constraint_satisfaction_{title.lower().replace(' ', '_')}.png"
        # )
    plt.tight_layout()
    plt.savefig(figures_dir / "constraint_satisfaction.png")

    # Create table for success rate with tolerance of 1e-3.
    for df, _ in df_and_title:
        df["overall_success"] = (
            (df["agent_obstacle_max_residual"] < 1e-3)
            & (df["agent_agent_max_residual"] < 1e-3)
            & (df["velocity_max_residual"] < 1e-3)
        )

    for key in ["overall_success", "solve_time"]:
        # Generate table of solve times, in LaTeX.
        num_agents = set()
        for df, _ in df_and_title:
            num_agents.update(df["num_robots"].unique())
        table_data = {
            "Problem Set": [t for (d, t) in df_and_title],
            **{
                f"{n} agents": [
                    d[d["num_robots"] == n][key].mean() for (d, t) in df_and_title
                ]
                for n in sorted(num_agents)
            },
        }

        pd.DataFrame(table_data).fillna("-").to_latex(
            figures_dir / f"average_{key}.tex", index=False
        )


if __name__ == "__main__":
    main()
