#!/usr/bin/env python
import gymnasium as gym

import rlp


class PuzzleRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # not a meaningful example, but shows how to use
        # the internal puzzle state for intermediate rewards
        reward -= info["puzzle_state"]["w"]

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = gym.make("rlp/Puzzle-v0", puzzle="fifteen", render_mode="human", params="2x2")
    wrapped_env = PuzzleRewardWrapper(env)

    observation, info = wrapped_env.reset(seed=42)
    for step in range(10000):
        action = env.action_space.sample()  # a random policy
        observation, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Step {step}: Action {action}, Reward: {reward}")

        if terminated or truncated:
            observation, info = wrapped_env.reset()
            print(f"Resetting")
wrapped_env.close()
