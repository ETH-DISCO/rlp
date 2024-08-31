#!/usr/bin/env python
import os

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit, FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import QRDQN, RecurrentPPO, MaskablePPO, TRPO

import rlp


if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument(
        "-t", "--timesteps", type=int, help="Number of timesteps during training"
    )
    parser.add_argument("-r", "--numrun", type=int, help="Training Run Number")
    parser.add_argument(
        "-alg",
        "--algorithm",
        type=str,
        help="Choice of RL Algorithm used for training",
        choices=[
            "PPO",
            "DQN",
            "A2C",
            "HER",
            "ARS",
            "QRDQN",
            "RecurrentPPO",
            "TRPO",
            "MaskablePPO",
        ],
    )
    parser.add_argument(
        "-ot",
        "--obs-type",
        type=str,
        help="Type of observation",
        choices=["rgb", "puzzle_state"],
        default="puzzle_state",
    )
    args = parser.parse_args()

    data_dir = f"/tmp/rlp/monitor/"

    if args.allowundo:
        undo_prefix = "undo"
    else:
        undo_prefix = "noundo"

    log_dir = f"{data_dir}trained_runs/{args.algorithm}_{args.timesteps}/{args.puzzle}_{args.arg}_{undo_prefix}_{args.obs_type}/"
    os.makedirs(log_dir, exist_ok=True)

    render_mode = "human" if not args.headless else "rgb_array"
    allow_undo = True if args.allowundo else False

    print(f"log_dir = {log_dir}")
    env_kwargs = dict(
        puzzle=args.puzzle,
        params=args.arg,
        render_mode=render_mode,
        window_width=128,
        window_height=128,
        allow_undo=args.allowundo,
        max_state_repeats=10000,
        include_cursor_in_state_info=True,
        obs_type=args.obs_type,
    )

    env = gym.make("rlp/Puzzle-v0", None, None, None, **env_kwargs)
    if args.obs_type == "puzzle_state":
        env = FlattenObservation(env)
    model_prefix = "best"
    model_suffix = f"_{args.puzzle}"

    env = Monitor(env, log_dir, override_existing=True)
    max_timesteps = 10000
    env = TimeLimit(env, max_timesteps)

    model_file = f"{data_dir}{args.algorithm}_{args.timesteps}/{args.puzzle}_{args.arg}_{undo_prefix}_{args.obs_type}/best_model{model_suffix}"
    print(f"Loading model {model_file}")

    buffer_size = 1000000
    model: PPO | DQN | A2C | QRDQN | RecurrentPPO | TRPO | MaskablePPO
    if args.algorithm == "PPO":
        model = PPO.load(model_file, env=env)
    elif args.algorithm == "DQN":
        model = DQN.load(model_file, env=env)
    elif args.algorithm == "A2C":
        model = A2C.load(model_file, env=env)
    elif args.algorithm == "QRDQN":
        model = QRDQN.load(model_file, env=env)
    elif args.algorithm == "RecurrentPPO":
        model = RecurrentPPO.load(model_file, env=env)
    elif args.algorithm == "TRPO":
        model = TRPO.load(model_file, env=env)
    elif args.algorithm == "MaskablePPO":
        model = MaskablePPO.load(model_file, env=env)
    else:
        raise Exception(f"{args.algorithm} is not supported")
    episodes = 1000
    timesteps = 0

    obs, info = env.reset()

    res = evaluate_policy(model, env, n_eval_episodes=50, render=True, deterministic=False, return_episode_rewards=True)

    print(f"Mean reward: {np.mean(res[0])}, std: {np.std(res[0])}")
    print(f"Mean episode length: {np.mean(res[1])}, std: {np.std(res[1])}", flush=True)
    print(f"Number of puzzles solved: {np.sum(np.array(res[0]) > 0)} out of {len(res[0])}", flush=True)

    while episodes > 0:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        timesteps += 1

        if terminated or truncated:
            episodes -= 1
            obs, info = env.reset()
            if terminated:
                print(
                    f"episode {1000-episodes} terminated after {timesteps} steps",
                    flush=True,
                )
            if truncated:
                print(
                    f"episode {1000-episodes} truncated after {timesteps} steps",
                    flush=True,
                )
            timesteps = 0
    env.close()
