from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, overload

import numpy as np
import torch

TensorType = TypeVar("TensorType", torch.Tensor, np.ndarray)


@dataclass
class Obstacles:
    positions: torch.Tensor
    radii: torch.Tensor


@dataclass
class SolutionValue:
    agent_agent_distances: torch.Tensor
    agent_obstacle_distances: torch.Tensor
    agent_positions: torch.Tensor

    def get_batch_item(self, index: int):
        return SolutionValue(
            agent_positions=self.agent_positions[index],
            agent_agent_distances=self.agent_agent_distances[index],
            agent_obstacle_distances=self.agent_obstacle_distances[index],
        )


@overload
def _tensor(array, type: Literal["torch"]) -> torch.Tensor: ...


@overload
def _tensor(array, type: Literal["numpy"]) -> np.ndarray: ...


def _tensor(array, type: Literal["torch", "numpy"] = "torch"):
    match type:
        case "torch":
            return torch.tensor(array, dtype=torch.float32)
        case "numpy":
            return np.array(array, dtype=np.float32)
        case _:
            raise ValueError(f"Unknown type: {type}")


@dataclass
class Problem(Generic[TensorType]):
    num_timesteps: int
    agent_start_positions: TensorType
    agent_end_positions: TensorType
    agent_reference_trajectory: TensorType | None
    agent_radii: TensorType
    agent_max_speeds: TensorType

    # These fields are for the 'map'.
    circular_obstacle_positions: TensorType
    circular_obstacle_radii: TensorType
    axis_aligned_box_obstacle_bounds: TensorType

    identifier: str | None = None

    @property
    def num_agents(self):
        return self.agent_start_positions.shape[0]

    @property
    def num_circular_obstacles(self):
        return self.circular_obstacle_positions.shape[0]

    @staticmethod
    def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(array, np.ndarray):
            return array

        return array.detach().cpu().numpy()

    @overload
    @classmethod
    def from_json(cls, entry) -> "Problem[torch.Tensor]": ...

    @overload
    @classmethod
    def from_json(cls, entry, type: Literal["torch"]) -> "Problem[torch.Tensor]": ...

    @overload
    @classmethod
    def from_json(cls, entry, type: Literal["numpy"]) -> "Problem[np.ndarray]": ...

    @classmethod
    def from_json(cls, entry, type: Literal["torch", "numpy"] = "numpy") -> Any:
        return cast(Any, cls)(
            num_timesteps=entry["num_timesteps"],
            agent_start_positions=_tensor(
                entry["agents"]["start_positions"], type=type
            ),
            agent_end_positions=_tensor(entry["agents"]["end_positions"], type=type),
            agent_radii=_tensor(entry["agents"]["radii"], type=type),
            agent_max_speeds=_tensor(entry["agents"]["max_speeds"], type=type),
            agent_reference_trajectory=None,
            obstacle_positions=_tensor(entry["obstacles"]["positions"], type=type),
            obstacle_radii=_tensor(entry["obstacles"]["radii"], type=type),
            identifier=f"sample_{entry['sample_idx']}"
            if "sample_idx" in entry
            else None,
        )
