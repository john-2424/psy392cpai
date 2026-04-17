from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded


@dataclass(frozen=True)
class Layout:
    start: tuple[int, int]
    goal: tuple[int, int]
    walls: set[tuple[int, int]]


class GridWorldEnv(EnvBase):
    batch_locked = False

    ACTION_TO_DELTA = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }

    # Channel permutation for obs_remap: original [agent, goal, walls]
    # becomes [goal, walls, agent] -- same information present but the
    # conv filters trained on the stable mapping see shuffled semantics.
    # Analog of hippocampal global remapping (Sanders et al. 2020).
    OBS_REMAP_PERM = (1, 2, 0)

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 50,
        change_mode: str = "stable",
        observation_mode: str = "normal",
        device: str | torch.device = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(device=device, batch_size=[])

        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.change_mode = str(change_mode)
        if observation_mode not in {"normal", "visual_perturb", "obs_remap"}:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
        self.observation_mode = observation_mode

        self.step_count = 0
        self.agent_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.walls: set[tuple[int, int]] = set()

        self.rng = torch.Generator(device="cpu")
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._set_seed(seed)

        # Deterministic distractor mask for visual_perturb: 10% of cells
        # get extra 0.3-intensity pixels on a fixed overlay baked into the
        # agent channel. Same perturbation across episodes for a given
        # env seed, different across env instances.
        self._distractor = self._sample_distractor_mask()

        self._make_specs()

    def _sample_distractor_mask(self) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 9973)
        total = self.grid_size * self.grid_size
        mask = torch.zeros(total, dtype=torch.float32)
        n_distract = max(1, total // 10)
        idx = torch.randperm(total, generator=gen)[:n_distract]
        mask[idx] = 0.3
        return mask.view(self.grid_size, self.grid_size).to(self.device)

    def _make_specs(self) -> None:
        obs_shape = torch.Size([3, self.grid_size, self.grid_size])

        self.observation_spec = Composite(
            observation=Bounded(
                low=0.0,
                high=1.0,
                shape=obs_shape,
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
        )

        self.action_spec = Categorical(
            n=4,
            shape=torch.Size([]),
            dtype=torch.int64,
            device=self.device,
        )

        self.reward_spec = Unbounded(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = Composite(
            done=Bounded(
                low=0,
                high=1,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Bounded(
                low=0,
                high=1,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=Bounded(
                low=0,
                high=1,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=torch.Size([]),
        )

    def _get_layout(self, mode: str) -> Layout:
        # Stable: gap at (4, 3)
        # stable_walls = {(1, 3), (2, 3), (3, 3), (5, 3), (6, 3)}
        stable_walls = {(2, 3), (3, 3), (5, 3)}
        # Transition change: gap moved to (2, 3)
        # transition_walls = {(1, 3), (3, 3), (4, 3), (5, 3), (6, 3)}
        transition_walls = {(2, 3), (4, 3), (5, 3)}

        if mode == "stable":
            return Layout(start=(6, 1), goal=(1, 6), walls=stable_walls)
        if mode == "reward_change":
            return Layout(start=(6, 1), goal=(1, 1), walls=stable_walls)
        if mode == "transition_change":
            return Layout(start=(6, 1), goal=(1, 6), walls=transition_walls)

        raise ValueError(f"Unknown change_mode: {mode}")

    def _build_observation(self) -> torch.Tensor:
        obs = torch.zeros(
            (3, self.grid_size, self.grid_size),
            dtype=torch.float32,
            device=self.device,
        )

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        obs[0, ar, ac] = 1.0
        obs[1, gr, gc] = 1.0

        for wr, wc in self.walls:
            obs[2, wr, wc] = 1.0

        if self.observation_mode == "visual_perturb":
            # Distractor pixels layered onto the agent channel. State
            # identity unchanged (agent marker still 1.0), but input
            # statistics shift. Clamp to [0, 1] to honour the obs spec.
            obs[0] = torch.clamp(obs[0] + self._distractor, 0.0, 1.0)
        elif self.observation_mode == "obs_remap":
            perm = list(self.OBS_REMAP_PERM)
            obs = obs[perm]

        return obs

    def _proposed_next_pos(self, action: int) -> tuple[int, int]:
        dr, dc = self.ACTION_TO_DELTA[action]
        r, c = self.agent_pos
        return (r + dr, c + dc)

    def _is_valid(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        in_bounds = 0 <= r < self.grid_size and 0 <= c < self.grid_size
        not_wall = pos not in self.walls
        return in_bounds and not_wall

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        self.step_count = 0
        layout = self._get_layout(self.change_mode)

        self.agent_pos = layout.start
        self.goal_pos = layout.goal
        self.walls = set(layout.walls)

        obs = self._build_observation()
        return TensorDict(
            {"observation": obs},
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = int(tensordict["action"].item())
        self.step_count += 1

        next_pos = self._proposed_next_pos(action)
        if self._is_valid(next_pos):
            self.agent_pos = next_pos

        success = self.agent_pos == self.goal_pos
        terminated = success
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        reward = 1.0 if success else -0.01
        obs = self._build_observation()

        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

    def _set_seed(self, seed: Optional[int]) -> int:
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.rng.manual_seed(int(seed))
        self.seed = int(seed)
        return self.seed

    def set_change_mode(self, mode: str) -> None:
        self.change_mode = mode

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r, c in self.walls:
            grid[r][c] = "#"
        gr, gc = self.goal_pos
        ar, ac = self.agent_pos
        grid[gr][gc] = "G"
        grid[ar][ac] = "A"
        return "\n".join(" ".join(row) for row in grid)