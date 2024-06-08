import re

import embodied
import numpy as np
import os
import json

from dreamerv3.embodied import Counter


def eval_only(agent, env, logger, args, nr_eval_episodes=None):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_log = embodied.when.Clock(args.log_every)
    step = logger.step
    nr_episodes = Counter()
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    all_rewards = []
    all_episode_lengths = []

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        stats["length"] = length
        all_episode_lengths.append(length)
        stats["score"] = score
        all_rewards.append(score)
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")
        nr_episodes.increment()

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    if nr_eval_episodes is None:
        while step < args.steps:
            driver(policy, steps=100)
            if should_log(step):
                logger.add(metrics.result())
                logger.add(timer.stats(), prefix="timer")
                logger.write(fps=True)
        logger.write()
    else:
        while nr_episodes < nr_eval_episodes:
            driver(policy, episodes=1)
        # write results from metrics to json in logdir
        # result = metrics.result()
        all_rewards = np.array(all_rewards)
        all_episode_lengths = np.array(all_episode_lengths)

        # also store these arrays in a file
        np.save(os.path.join(args.logdir, "all_rewards.npy"), all_rewards)
        np.save(
            os.path.join(args.logdir, "all_episode_lengths.npy"), all_episode_lengths
        )

        # Calculate average and standard deviation
        average_reward = np.mean(all_rewards)
        average_episode_length = np.mean(all_episode_lengths)
        average_success_episode_length = np.mean(all_episode_lengths[all_rewards > 0])
        std_dev_episode_length = np.std(all_episode_lengths)
        success_rate = np.where(all_rewards > 0, 1.0, 0.0).mean()

        # Save results to a JSON file
        results_data = {
            "Average Reward": average_reward,
            "Average Episode Length": average_episode_length,
            "Average Success Episode Length": average_success_episode_length,
            "Standard Deviation of Episode Length": std_dev_episode_length,
            "Success Rate": success_rate,
        }
        results_file_path = os.path.join(args.logdir, "evaluation_results.json")
        with open(results_file_path, "w") as file:
            json.dump(results_data, file, indent=4)
