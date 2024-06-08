#!/usr/bin/env python
import os
import fnmatch
from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym
import pygame
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import TimeAwareObservation

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


from wandb.integration.sb3 import WandbCallback
import wandb

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

import rlp

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        data_dim,
        embedding_dim,
        nhead,
        num_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(TransformerFeaturesExtractor, self).__init__(
            observation_space, embedding_dim
        )
        self.transformer = Transformer(
            embedding_dim=embedding_dim,
            data_dim=data_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        # Extract the 'obs' key from the dict
        obs = observations["obs"]
        length = observations["len"]
        # all elements of length should be the same (we can't train on different puzzle sizes at the same time)
        length = int(length[0])
        obs = obs[:, :length]
        # Return the embedding of the cursor token (which is last)
        return self.transformer(obs)[:, -1, :]


class Transformer(nn.Module):
    def __init__(
        self, embedding_dim, data_dim, nhead, num_layers, dim_feedforward, dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.data_dim = data_dim

        self.lin = nn.Linear(data_dim, embedding_dim)

        encoder_layers = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        # x is of shape (batch_size, seq_length, embedding_dim)
        x = self.lin(x)
        transformed = self.transformer_encoder(x)
        return transformed


class EvalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EvalWrapper, self).__init__(env)
        self.t = 0

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.t += 1
        observation, reward, terminated, truncated, info = self.env.step(action)
        # We can take a large number of max steps just to be safe.
        if reward > 0:
            reward += 100000 - self.t
        if "episode" in info:
            info["episode"]["r"] = reward
            info["episode"]["l"] = self.t
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)


class NetslideTransformerWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(NetslideTransformerWrapper, self).__init__(env)
        self.original_space = env.observation_space
        # The original observation is an ordereddict with the keys ['barriers', 'cursor_pos', 'height',
        # 'last_move_col', 'last_move_dir', 'last_move_row', 'move_count', 'movetarget', 'tiles', 'width', 'wrapping']
        self.max_length = 512
        self.embedding_dim = 16 + 4
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.max_length,
                self.embedding_dim,
            ),
            dtype=np.float32,
        )
        # wrap the whole thing in a dict because of reasons
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.observation_space,
                "len": gym.spaces.Box(
                    low=0, high=self.max_length, shape=(1,), dtype=np.int32
                ),
            }
        )

    def observation(self, obs):
        barriers = obs["barriers"]
        # each element of barriers is an uint16. We can convert it to a binary array
        barriers = np.unpackbits(barriers.view(np.uint8)).reshape(-1, 16)
        # add some positional embedding to the barriers
        embedded_barriers = np.concatenate(
            [
                barriers,
                self.pos_embedding(
                    np.arange(barriers.shape[0]), obs["width"], obs["height"]
                ),
            ],
            axis=1,
        )
        tiles = obs["tiles"]
        # each element of tiles is an uint16. We can convert it to a binary array
        tiles = np.unpackbits(tiles.view(np.uint8)).reshape(-1, 16)
        # add some positional embedding to the tiles
        embedded_tiles = np.concatenate(
            [
                tiles,
                self.pos_embedding(
                    np.arange(tiles.shape[0]), obs["width"], obs["height"]
                ),
            ],
            axis=1,
        )
        cursor_pos = obs["cursor_pos"]
        # this is 2d, lets just repeat
        embedded_cursor_pos = np.concatenate(
            [
                np.ones((1, 16)),
                self.pos_embedding_cursor(cursor_pos, obs["width"], obs["height"]),
            ],
            axis=1,
        )
        # let's concatenate all the embeddings
        embedded_obs = np.concatenate(
            [embedded_barriers, embedded_tiles, embedded_cursor_pos], axis=0
        )
        # we need to return a dict with obs in the obs key
        current_length = embedded_obs.shape[0]
        # pad with zeros
        if current_length < self.max_length:
            embedded_obs = np.concatenate(
                [
                    embedded_obs,
                    np.zeros((self.max_length - current_length, self.embedding_dim)),
                ],
                axis=0,
            )
        return {"obs": embedded_obs, "len": np.array([current_length])}

    @staticmethod
    def pos_embedding(pos, width, height):
        # pos is an array of integers from 0 to width*height
        # width and height are integers
        # return a 2D array with the positional embedding, using sin and cos
        # the embedding should have shape (width*height, self.embedding_dim)
        x, y = pos % width, pos // width
        # x and y are integers from 0 to width-1 and height-1
        pos_embed = np.zeros((len(pos), 4))
        pos_embed[:, 0] = np.sin(2 * np.pi * x / width)
        pos_embed[:, 1] = np.cos(2 * np.pi * x / width)
        pos_embed[:, 2] = np.sin(2 * np.pi * y / height)
        pos_embed[:, 3] = np.cos(2 * np.pi * y / height)
        return pos_embed

    @staticmethod
    def pos_embedding_cursor(pos, width, height):
        # cursor pos somehow goes from -1 to width or height
        # lets adjust for this
        x, y = pos
        x += 1
        y += 1
        width += 1
        height += 1
        pos_embed = np.zeros((1, 4))
        pos_embed[0, 0] = np.sin(2 * np.pi * x / width)
        pos_embed[0, 1] = np.cos(2 * np.pi * x / width)
        pos_embed[0, 2] = np.sin(2 * np.pi * y / height)
        pos_embed[0, 3] = np.cos(2 * np.pi * y / height)
        return pos_embed


class TentsTransformerWrapper(NetslideTransformerWrapper):
    def __init__(self, env):
        super(TentsTransformerWrapper, self).__init__(env)
        self.original_space = env.observation_space
        # The original observation is an ordereddict with the keys ['barriers', 'cursor_pos', 'height', 'last_move_col', 'last_move_dir', 'last_move_row', 'move_count', 'movetarget', 'tiles', 'width', 'wrapping']
        self.max_length = 512
        self.embedding_dim = 4 + 4
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.max_length,
                self.embedding_dim,
            ),
            dtype=np.float32,
        )
        # wrap the whole thing in a dict because of reasons
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.observation_space,
                "len": gym.spaces.Box(
                    low=0, high=self.max_length, shape=(1,), dtype=np.int32
                ),
            }
        )

    def observation(self, obs):
        grid = obs["grid"]
        # each element of the grid contains values 0 to 4. We can convert it to a binary array
        grid = np.unpackbits(grid.astype(np.uint8)).reshape(-1, 8)[:, -4:]
        # add some positional embedding to the tiles
        embedded_grid = np.concatenate(
            [grid, self.pos_embedding(np.arange(grid.shape[0]), obs["w"], obs["h"])],
            axis=1,
        )

        cursor_pos = obs["cursor_pos"]
        # this is 2d, lets just repeat
        embedded_cursor_pos = np.concatenate(
            [
                np.ones((1, 4)),
                self.pos_embedding_cursor(cursor_pos, obs["w"], obs["h"]),
            ],
            axis=1,
        )

        numbers = obs["numbers"].astype(np.float32)
        # this has length w + h
        max_nr = np.ceil(np.max((obs["w"], obs["h"])) / 2)
        numbers /= max_nr
        # numbers is currently 1d, expand it to 2d like the others
        numbers = np.concatenate(
            (
                np.ones((len(numbers), 1)),
                np.zeros((len(numbers), 2)),
                np.expand_dims(numbers, axis=1),
            ),
            axis=-1,
        )
        embedded_numbers = np.concatenate(
            [numbers, self.pos_embedding_numbers(obs["w"], obs["h"])], axis=1
        )

        # let's concatenate all the embeddings
        embedded_obs = np.concatenate(
            [embedded_grid, embedded_cursor_pos, embedded_numbers], axis=0
        )
        # we need to return a dict with obs in the obs key
        current_length = embedded_obs.shape[0]
        # pad with zeros
        if current_length < self.max_length:
            embedded_obs = np.concatenate(
                [
                    embedded_obs,
                    np.zeros((self.max_length - current_length, self.embedding_dim)),
                ],
                axis=0,
            )
        return {"obs": embedded_obs, "len": np.array([current_length])}

    @staticmethod
    def pos_embedding_numbers(w, h):
        x = np.concatenate([np.arange(w), np.zeros(h)])
        y = np.concatenate([np.zeros(w), np.arange(h)])
        pos_embed = np.zeros((len(x), 4))
        pos_embed[:, 0] = np.sin(2 * np.pi * x / w)
        pos_embed[:, 1] = np.cos(2 * np.pi * x / w)
        pos_embed[:, 2] = np.sin(2 * np.pi * y / h)
        pos_embed[:, 3] = np.cos(2 * np.pi * y / h)
        return pos_embed


class InertiaTransformerWrapper(NetslideTransformerWrapper):
    def __init__(self, env):
        super(InertiaTransformerWrapper, self).__init__(env)
        self.original_space = env.observation_space
        # The original observation is an ordereddict with the keys ['barriers', 'cursor_pos', 'height',
        # 'last_move_col', 'last_move_dir', 'last_move_row', 'move_count', 'movetarget', 'tiles', 'width', 'wrapping']
        self.max_length = 512
        self.embedding_dim = 8 + 4
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.max_length,
                self.embedding_dim,
            ),
            dtype=np.float32,
        )
        # wrap the whole thing in a dict because of reasons
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.observation_space,
                "len": gym.spaces.Box(
                    low=0, high=self.max_length, shape=(1,), dtype=np.int32
                ),
            }
        )

    def observation(self, obs):
        grid = obs["grid"]
        # each element of the grid contains values 0 to 4. We can convert it to a binary array
        grid = np.unpackbits(grid.astype(np.uint8)).reshape(-1, 8)[:, -5:]
        # add some positional embedding to the tiles
        embedded_grid = np.concatenate(
            [grid, self.pos_embedding(np.arange(grid.shape[0]), obs["w"], obs["h"])],
            axis=1,
        )

        cursor_pos = (obs["px"], obs["py"])
        # this is 2d, lets just repeat
        embedded_cursor_pos = np.concatenate(
            [
                np.ones((1, 5)),
                self.pos_embedding_cursor(cursor_pos, obs["w"], obs["h"]),
            ],
            axis=1,
        )

        # let's concatenate all the embeddings
        embedded_obs = np.concatenate([embedded_grid, embedded_cursor_pos], axis=0)
        # we need to return a dict with obs in the obs key
        current_length = embedded_obs.shape[0]
        # pad with zeros
        if current_length < self.max_length:
            embedded_obs = np.concatenate(
                [
                    embedded_obs,
                    np.zeros((self.max_length - current_length, self.embedding_dim)),
                ],
                axis=0,
            )
        return {"obs": embedded_obs, "len": np.array([current_length])}


class SameGameTransformerWrapper(NetslideTransformerWrapper):
    def __init__(self, env):
        super(SameGameTransformerWrapper, self).__init__(env)
        self.original_space = env.observation_space
        # The original observation is an ordereddict with the keys ['barriers', 'cursor_pos', 'height',
        # 'last_move_col', 'last_move_dir', 'last_move_row', 'move_count', 'movetarget', 'tiles', 'width', 'wrapping']
        self.max_length = 512
        self.embedding_dim = 8 + 4
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.max_length,
                self.embedding_dim,
            ),
            dtype=np.float32,
        )
        # wrap the whole thing in a dict because of reasons
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.observation_space,
                "len": gym.spaces.Box(
                    low=0, high=self.max_length, shape=(1,), dtype=np.int32
                ),
            }
        )

    def observation(self, obs):
        tiles = obs["tiles"]
        # each element of the grid contains values a maximum of 9 colors. We can convert it to a binary array
        tiles = np.unpackbits(tiles.astype(np.uint8)).reshape(-1, 8)
        # add some positional embedding to the tiles
        embedded_tiles = np.concatenate(
            [tiles, self.pos_embedding(np.arange(tiles.shape[0]), obs["w"], obs["h"])],
            axis=1,
        )

        selected_tiles = obs["selected_tiles"]
        selected_tiles = selected_tiles // 256
        selected_tiles = selected_tiles.astype(np.uint8)
        # each element of the grid contains values a maximum of 9 colors. We can convert it to a binary array
        selected_tiles = np.unpackbits(selected_tiles.astype(np.uint8)).reshape(-1, 8)
        selected_tiles[:, 0] = 1
        # add some positional embedding to the tiles
        embedded_selected_tiles = np.concatenate(
            [
                selected_tiles,
                self.pos_embedding(
                    np.arange(selected_tiles.shape[0]), obs["w"], obs["h"]
                ),
            ],
            axis=1,
        )

        cursor_pos = obs["cursor_pos"]
        # this is 2d, lets just repeat
        embedded_cursor_pos = np.concatenate(
            [
                np.ones((1, 8)),
                self.pos_embedding_cursor(cursor_pos, obs["w"], obs["h"]),
            ],
            axis=1,
        )

        # let's concatenate all the embeddings
        embedded_obs = np.concatenate(
            [embedded_tiles, embedded_selected_tiles, embedded_cursor_pos], axis=0
        )
        # we need to return a dict with obs in the obs key
        current_length = embedded_obs.shape[0]
        # pad with zeros
        if current_length < self.max_length:
            embedded_obs = np.concatenate(
                [
                    embedded_obs,
                    np.zeros((self.max_length - current_length, self.embedding_dim)),
                ],
                axis=0,
            )
        return {"obs": embedded_obs, "len": np.array([current_length])}


class UntangleTransformerWrapper(NetslideTransformerWrapper):
    def __init__(self, env):
        super(UntangleTransformerWrapper, self).__init__(env)
        self.original_space = env.observation_space
        # The original observation is an ordereddict with the keys ['barriers', 'cursor_pos', 'height',
        # 'last_move_col', 'last_move_dir', 'last_move_row', 'move_count', 'movetarget', 'tiles', 'width', 'wrapping']
        self.max_length = 512
        self.embedding_dim = 1 + 4
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.max_length,
                self.embedding_dim,
            ),
            dtype=np.float32,
        )
        # wrap the whole thing in a dict because of reasons
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.observation_space,
                "len": gym.spaces.Box(
                    low=0, high=self.max_length, shape=(1,), dtype=np.int32
                ),
            }
        )

    def observation(self, obs):
        pts_x = obs["pts"]["x"] / 256
        pts_y = obs["pts"]["y"] / 256
        denominator = obs["pts"]["d"] * 2
        edges = obs["edges"].reshape(-1, 2)

        res = []
        for edge in edges:
            if edge[0] == -1:
                continue
            res.append(
                np.array(
                    [
                        pts_x[edge[0]] / denominator[edge[0]],
                        pts_y[edge[0]] / denominator[edge[0]],
                        pts_x[edge[1]] / denominator[edge[1]],
                        pts_y[edge[1]] / denominator[edge[1]],
                    ]
                )
            )
        res = np.array(res)
        res = np.concatenate([np.zeros((res.shape[0], 1)), res], axis=1)

        cursor_pos = obs["cursor_pos"]
        # this is 2d, lets just repeat
        embedded_cursor_pos = np.concatenate(
            [np.ones((1, 1)), self.pos_embedding_cursor(cursor_pos, 84, 84)], axis=1
        )
        # let's concatenate all the embeddings
        embedded_obs = np.concatenate([res, embedded_cursor_pos], axis=0)
        # we need to return a dict with obs in the obs key
        current_length = embedded_obs.shape[0]
        # pad with zeros
        if current_length < self.max_length:
            embedded_obs = np.concatenate(
                [
                    embedded_obs,
                    np.zeros((self.max_length - current_length, self.embedding_dim)),
                ],
                axis=0,
            )
        return {"obs": embedded_obs, "len": np.array([current_length])}


class NegativeStepRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(NegativeStepRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == 0:
            return -0.01
        return reward


def wrapper_provider(env):
    if env == "netslide":
        return NetslideTransformerWrapper
    elif env == "tents":
        return TentsTransformerWrapper
    elif env == "inertia":
        return InertiaTransformerWrapper
    elif env == "samegame":
        return SameGameTransformerWrapper
    elif env == "untangle":
        return UntangleTransformerWrapper
    else:
        raise NotImplementedError(f"Unknown puzzle {env}")


data_dims = {
    "netslide": 20,
    "tents": 8,
    "inertia": 9,
    "samegame": 12,
    "untangle": 5,
}

extrapolation_args = {
    "netslide": {
        "2x3b1": "3x3b1",
    },
    "tents": {
        "4x4de": "5x5de",
    },
    "inertia": {
        "4x4": "5x5",
    },
    "samegame": {
        "2x3c3s2": "5x5c3s2",
    },
    "untangle": {
        "4": "6",
    },
}


if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument(
        "-t", "--timesteps", type=int, help="Number of timesteps during training"
    )
    parser.add_argument("-hn", "--hostname", type=str, help="hostname")
    parser.add_argument("-u", "--username", type=str, help="username")
    parser.add_argument("-n", "--numenvs", type=int, help="num of concurrent envs")
    parser.add_argument("-r", "--runnum", type=int, help="training run number")
    parser.add_argument(
        "-g", "--gamma", type=float, help="discount factor", default=0.99
    )
    parser.add_argument("-si", "--slurm_id", type=int, help="slurm id", default=0)
    parser.add_argument(
        "-me",
        "--max_episode_steps",
        type=int,
        help="max steps in single episode",
        default=10000,
    )
    parser.add_argument(
        "-rs",
        "--rollout_steps",
        type=int,
        help="#steps to run for each env per update",
        default=2048,
    )
    parser.add_argument(
        "-ec", "--entropy_coef", type=float, help="entropy coefficient", default=0.00
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, help="learning rate", default=3e-4
    )
    parser.add_argument(
        "-ne", "--num_epochs", type=int, help="number of epochs", default=10
    )
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=64)
    parser.add_argument(
        "-nr",
        "--negative_reward",
        action="store_true",
        help="use negative reward for steps",
    )
    parser.add_argument(
        "-ev",
        "--num_eval_episodes",
        type=int,
        help="number of episodes to evaluate",
        default=100,
    )
    parser.add_argument(
        "--transformer_ff_dim",
        type=int,
        help="transformer feed forward dimension",
        default=128,
    )
    parser.add_argument(
        "--transformer_dropout", type=float, help="transformer dropout", default=0.0
    )
    parser.add_argument(
        "--transformer_nhead", type=int, help="transformer number of heads", default=8
    )
    parser.add_argument(
        "--transformer_layers", type=int, help="transformer number of layers", default=3
    )
    parser.add_argument(
        "--transformer_embedding_dim",
        type=int,
        help="transformer embedding dimension",
        default=64,
    )
    parser.add_argument(
        "--test_on_extra", action="store_true", help="test on larger size"
    )

    args = parser.parse_args()

    data_dir = "./results/transformer/"

    log_dir = f"{data_dir}monitor/PPO_{args.timesteps}/{args.puzzle}_{args.arg}/run{args.runnum}/{args.slurm_id}"
    model_dir = f"{data_dir}models/PPO_{args.timesteps}/{args.puzzle}_{args.arg}/run{args.runnum}/{args.slurm_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"log_dir = {log_dir}", flush=True)
    print(f"model_dir = {model_dir}", flush=True)

    render_mode = "human" if not args.headless else "rgb_array"

    run = wandb.init(project="rlp", config=args, sync_tensorboard=True, dir=log_dir)

    puzzle_env_wrapper = wrapper_provider(args.puzzle)
    if args.negative_reward:
        env_wrapper_class = lambda env: NegativeStepRewardWrapper(
            puzzle_env_wrapper(env)
        )
    else:
        env_wrapper_class = puzzle_env_wrapper

    env = make_vec_env(
        "rlp/Puzzle-v0",
        env_kwargs=dict(
            puzzle=args.puzzle,
            render_mode=render_mode,
            params=args.arg,
            obs_type="puzzle_state",
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=args.numenvs,
        monitor_dir=log_dir,
        seed=args.runnum,
        wrapper_class=lambda env: Monitor(env_wrapper_class(env)),
        vec_env_cls=SubprocVecEnv,
    )
    eval_env_args = args.arg
    if args.test_on_extra:
        eval_env_args = extrapolation_args[args.puzzle][args.arg]
    eval_env = make_vec_env(
        "rlp/Puzzle-v0",
        env_kwargs=dict(
            puzzle=args.puzzle,
            render_mode=render_mode,
            params=eval_env_args,
            obs_type="puzzle_state",
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=1,
        monitor_dir=log_dir,
        seed=args.runnum,
        wrapper_class=lambda env: EvalWrapper(puzzle_env_wrapper(env)),
        vec_env_cls=SubprocVecEnv,
    )

    obs, info = env.reset()
    obs, info = eval_env.reset()

    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=20,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=100000,
        deterministic=False,
        render=False,
    )

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            embedding_dim=args.transformer_embedding_dim,
            nhead=args.transformer_nhead,
            num_layers=args.transformer_layers,
            dim_feedforward=args.transformer_ff_dim,
            dropout=args.transformer_dropout,
            data_dim=data_dims[args.puzzle],
        ),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=args.runnum,
        gamma=args.gamma,
        tensorboard_log=log_dir,
        n_steps=args.rollout_steps,
        ent_coef=args.entropy_coef,
        learning_rate=args.learning_rate,
        n_epochs=args.num_epochs,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[
            WandbCallback(),
            eval_callback,
        ],
        progress_bar=True,
    )

    # Eval best agent on same size
    eval_env = make_vec_env(
        "rlp/Puzzle-v0",
        env_kwargs=dict(
            puzzle=args.puzzle,
            render_mode=render_mode,
            params=args.arg,
            obs_type="puzzle_state",
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=1,
        monitor_dir=log_dir,
        seed=args.runnum,
        wrapper_class=lambda env: puzzle_env_wrapper(env),
    )
    obs, info = eval_env.reset()
    model = PPO.load(model_dir + "/best_model.zip")
    res = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.num_eval_episodes,
        render=False,
        deterministic=False,
        return_episode_rewards=True,
    )

    print(f"Mean reward: {np.mean(res[0])}, std: {np.std(res[0])}")
    print(f"Mean episode length: {np.mean(res[1])}, std: {np.std(res[1])}")

    # write results to wandb
    run.log(
        {
            "final_mean_reward": np.mean(res[0]),
            "final_std_reward": np.std(res[0]),
            "final_mean_episode_length": np.mean(res[1]),
            "final_std_episode_length": np.std(res[1]),
            "puzzles_solved": np.sum(np.array(res[0]) > 0),
        }
    )
    # write results to disk
    np.save(f"{log_dir}/final_rewards.npy", res[0])
    np.save(f"{log_dir}/final_episode_lengths.npy", res[1])

    new_difficulty = extrapolation_args[args.puzzle][args.arg]
    eval_env = make_vec_env(
        "rlp/Puzzle-v0",
        env_kwargs=dict(
            puzzle=args.puzzle,
            render_mode=render_mode,
            params=new_difficulty,
            obs_type="puzzle_state",
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=1,
        monitor_dir=log_dir,
        seed=args.runnum,
        wrapper_class=lambda env: puzzle_env_wrapper(env),
    )
    obs, info = eval_env.reset()
    model = PPO.load(model_dir + "/best_model.zip")
    res = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.num_eval_episodes,
        render=False,
        deterministic=False,
        return_episode_rewards=True,
    )

    print(f"Mean reward: {np.mean(res[0])}, std: {np.std(res[0])}")
    print(f"Mean episode length: {np.mean(res[1])}, std: {np.std(res[1])}", flush=True)

    # write results to wandb
    run.log(
        {
            "extra_final_mean_reward": np.mean(res[0]),
            "extra_final_std_reward": np.std(res[0]),
            "extra_final_mean_episode_length": np.mean(res[1]),
            "extra_final_std_episode_length": np.std(res[1]),
            "extra_puzzles_solved": np.sum(np.array(res[0]) > 0),
        }
    )
    # write results to disk
    np.save(f"{log_dir}/extra_final_rewards.npy", res[0])
    np.save(f"{log_dir}/extra_final_episode_lengths.npy", res[1])

    # close wandb run
    wandb.finish()

    os._exit(0)
