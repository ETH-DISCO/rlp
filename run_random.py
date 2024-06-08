#!/usr/bin/env python
import gymnasium as gym
import pygame

import rlp

if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    args = parser.parse_args()

    render_mode = "human" if not args.headless else "rgb_array"

    env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=args.puzzle,
        render_mode=render_mode,
        params=args.arg,
        window_width=512,
        window_height=512,
        obs_type="puzzle_state",
        include_cursor_in_state_info=True,
    )

    observation, info = env.reset(seed=42)
    for _ in range(100):
        action = env.action_space.sample()  # currently a random policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            print(f"Episode complete, resetting")
    env.close()
