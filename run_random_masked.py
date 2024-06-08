#!/usr/bin/env python
import gymnasium as gym
import pygame
import numpy as np

import rlp
from rlp import specific_api


if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    args = parser.parse_args()

    render_mode = "human" if not args.headless else "rgb_array"
    allow_undo = True if args.allowundo else False

    env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=args.puzzle,
        render_mode=render_mode,
        params=args.arg,
        window_width=512,
        window_height=512,
        allow_undo=allow_undo,
        obs_type="puzzle_state",
        include_cursor_in_state_info=True,
        max_state_repeats=1000,
    )

    observation, info = env.reset()

    # for printing the chosen action in readable form
    action_space = specific_api.get_action_keys(args.puzzle, allow_undo)
    action_strings = [
        pygame.key.name(action_space[s]) for s in range(len(action_space))
    ]

    for _ in range(10000):
        print()
        if isinstance(env, gym.Wrapper):
            mask = env.action_masks()
            if not np.any(mask):
                print(f"Agent has no more valid moves, resetting episode")
                observation, info = env.reset()
                continue
            print(f"mask {mask}")
        else:
            mask = None
        action = env.action_space.sample(mask=mask)  # currently a random policy
        print(f"Next action: [{action} -> {action_strings[action]}]")
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode complete, resetting")
            observation, info = env.reset()
    env.close()
