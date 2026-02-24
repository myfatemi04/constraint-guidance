"""For generating scientific figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_constraint_satisfaction_figure(
    df: pd.DataFrame, title: str, tolerance_levels=[1e-2, 1e-3, 1e-4, 1e-5]
):
    plt.figure(figsize=(12, 4))

    # 3 rows, 3 columns: row == num_robots, col == constraint type.
    for i_num_robots, num_robots in enumerate([3, 6, 9]):
        plt.subplot(1, 3, 1 + i_num_robots)
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

            plt.plot(
                tolerance_levels, tolerance_results, marker="o", label=constraint_type
            )

        plt.xscale("log")
        plt.yscale("linear")
        plt.xlabel("Constraint Satisfaction Tolerance")
        plt.ylabel("Fraction of Problems Satisfied")
        plt.title(f"Number of Robots: {num_robots}")
        plt.ylim(0, 1.05)
        plt.grid()
        plt.legend()

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


if __name__ == "__main__":
    # without Voronoi initialization
    paths_0 = {
        "dense": "results/2026-01-29/experiment_16-18-37_dense_num_robots=any/table.csv",
        "connected_room": "results/2026-01-30/experiment_13-32-20_connected_room_num_robots=any/table.csv",
        "shelf": "results/2026-01-30/experiment_13-27-10_shelf_num_robots=any/table.csv",
        "simple": "results/2026-01-29/experiment_16-44-37_simple_num_robots=any/table.csv",
    }
    # with Voronoi initialization
    paths_1 = {
        "connected_room": "results/2026-02-24/experiment_07-22-43_connected_room_num_robots=any/table.csv",
        "dense": "results/2026-02-24/experiment_07-31-38_dense_num_robots=any/table.csv",
        "shelf": "results/2026-02-24/experiment_07-36-12_shelf_num_robots=any/table.csv",
        "simple": "results/2026-02-24/experiment_07-39-32_simple_num_robots=any/table.csv",
    }
    # with Voronoi initialization and better schedule
    paths_2 = paths_1 | {
        k: f"results/2026-02-24/experiment_08-04-06/{k}/table.csv"
        for k in ["dense", "simple", "shelf", "connected_room"]
    }

    paths = paths_2

    dense_df = pd.read_csv(paths["dense"])
    connected_room_df = pd.read_csv(paths["connected_room"])
    shelf_df = pd.read_csv(paths["shelf"])
    simple_df = pd.read_csv(paths["simple"])
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    df_and_title = [
        (dense_df, "Dense"),
        (connected_room_df, "Connected Room"),
        (shelf_df, "Shelf"),
        (simple_df, "Simple"),
    ]

    for df, title in df_and_title:
        generate_constraint_satisfaction_figure(df, title)
        plt.savefig(
            figures_dir
            / f"constraint_satisfaction_{title.lower().replace(' ', '_')}.png"
        )

    # Create table for success rate with tolerance of 1e-3.
    for df, _ in df_and_title:
        df["overall_success"] = (
            (df["agent_obstacle_max_residual"] < 1e-3)
            & (df["agent_agent_max_residual"] < 1e-3)
            & (df["velocity_max_residual"] < 1e-3)
        )

    for key in ["overall_success", "solve_time"]:
        # Generate table of solve times, in LaTeX.
        table_data = {
            "Problem Set": [t for (d, t) in df_and_title],
            "3 agents": [
                d[d["num_robots"] == 3][key].mean() for (d, t) in df_and_title
            ],
            "6 agents": [
                d[d["num_robots"] == 6][key].mean() for (d, t) in df_and_title
            ],
            "9 agents": [
                d[d["num_robots"] == 9][key].mean() for (d, t) in df_and_title
            ],
        }

        pd.DataFrame(table_data).to_latex(
            figures_dir / f"average_{key}.tex", index=False
        )
