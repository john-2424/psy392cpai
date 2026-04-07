from __future__ import annotations

import torch
from torchrl.envs.utils import check_env_specs

from src.envs.gridworld import GridWorldEnv


def summarize_env(env: GridWorldEnv, mode: str) -> None:
    print(f"\n=== Mode: {mode} ===")
    td = env.reset()
    obs = td["observation"]

    print("Observation shape:", tuple(obs.shape))
    print("Agent:", env.agent_pos)
    print("Goal:", env.goal_pos)
    print("Walls:", sorted(env.walls))
    print(
        "Channel sums:",
        obs[0].sum().item(),
        obs[1].sum().item(),
        obs[2].sum().item(),
    )

    td["action"] = env.action_spec.rand()
    print("Random action:", td["action"])

    step_td = env.step(td)
    print("Step output keys:", step_td.keys(True))
    print(step_td)
    print("New agent position:", env.agent_pos)


def main() -> None:
    torch.manual_seed(0)
    modes = ["stable", "reward_change", "transition_change"]

    for mode in modes:
        env = GridWorldEnv(change_mode=mode, seed=0)
        print(f"\n######## Checking mode: {mode} ########")
        # print(env.observation_spec)
        # print(env.action_spec)
        # print(env.reward_spec)
        # print(env.done_spec)
        check_env_specs(env)
        print("check_env_specs: PASSED")
        summarize_env(env, mode)

    stable_env = GridWorldEnv(change_mode="stable", seed=0)
    reward_env = GridWorldEnv(change_mode="reward_change", seed=0)
    transition_env = GridWorldEnv(change_mode="transition_change", seed=0)

    stable_env.reset()
    reward_env.reset()
    transition_env.reset()

    assert stable_env.walls == reward_env.walls
    assert stable_env.goal_pos != reward_env.goal_pos
    assert stable_env.goal_pos == transition_env.goal_pos
    assert stable_env.walls != transition_env.walls

    print("\nAll environment checks passed.")


if __name__ == "__main__":
    main()