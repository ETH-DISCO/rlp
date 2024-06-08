import ray
import numpy
import json
import os

import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time
import pathlib
from math import ceil

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer
from functools import partial


@ray.remote
class TestWorker:
    def __init__(self, checkpoint, Game, config, num_gpus):
        self.self_play_worker = self_play.SelfPlay.options(
            num_cpus=0, num_gpus=num_gpus
        ).remote(checkpoint, Game, config, numpy.random.randint(10000))

    async def play_game(self, render, opponent, muzero_player):
        return await self.self_play_worker.play_game.remote(
            0, 0, render, opponent, muzero_player
        )


def test_multihread(
    self,
    render=True,
    opponent=None,
    muzero_player=None,
    num_tests=1,
    num_gpus=0,
    num_workers=4,
    results_dir="./results",
):
    opponent = opponent if opponent else self.config.opponent
    muzero_player = muzero_player if muzero_player else self.config.muzero_player
    ray.init(ignore_reinit_error=True)

    # Create TestWorker instances
    workers = [
        TestWorker.remote(self.checkpoint, self.Game, self.config, num_gpus)
        for _ in range(num_workers)
    ]

    results = []
    futures = [
        worker.play_game.remote(render, opponent, muzero_player) for worker in workers
    ]
    while len(results) < num_tests:
        done_id, futures = ray.wait(futures)
        result = ray.get(done_id[0])
        results.append(result)
        if len(results) < num_tests:
            # Add a new game to the pool
            futures.extend(
                [
                    workers[done_id[0] % num_workers].play_game.remote(
                        render, opponent, muzero_player
                    )
                ]
            )

    # Process results
    if len(self.config.players) == 1:
        total_reward = numpy.mean([sum(history.reward_history) for history in results])
    else:
        total_reward = numpy.mean(
            [
                sum(
                    reward
                    for i, reward in enumerate(history.reward_history)
                    if history.to_play_history[i - 1] == muzero_player
                )
                for history in results
            ]
        )

    average_episode_length = numpy.mean(
        [len(history.action_history) - 1 for history in results]
    )

    # Save results to a JSON file
    results_data = {
        "Average Reward": total_reward,
        "Average Episode Length": average_episode_length,
    }
    results_file_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_file_path, "w") as file:
        json.dump(results_data, file, indent=4)

    print(
        f"\nAverage reward: {total_reward}, Average episode length: {average_episode_length}\n"
    )
    return total_reward


if __name__ == "__main__":
    if len(sys.argv) == 5:
        config = json.loads(sys.argv[2])
        muzero = MuZero(sys.argv[1], config)
        muzero.load_model(sys.argv[3])
        test(render=False, num_tests=int(sys.argv[4]), num_workers=32)
    else:
        print("Usage: python muzero.py <game> <config> <model> <num_tests>")
