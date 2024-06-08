import os.path

import rlp
import sys


def main():

    import warnings
    import dreamerv3
    import gymnasium as gym
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    if sys.argv[1] == "test":
        config = embodied.Config()
        config = config.load(os.path.join(sys.argv[2], "config.yaml"))

        max_env_steps = 10000
        max_state_repeats = max_env_steps

        puzzle = sys.argv[3]
        params = sys.argv[4]
        seed = int(sys.argv[5])
        test = True

        # if config.logdir looks something like .../puzzle_name/params/seed, then update config.logdir to .../puzzle_name/params/10000/10000/seed
        # check if this is the case by checking if the last three directories are numbers
        logdir = config.logdir
        logdir_split = logdir.split("/")
        if (
            logdir_split[-1].isdigit()
            and logdir_split[-2].isdigit()
            and logdir_split[-3].isdigit()
        ):
            pass
        else:
            # we need to add the max_env_steps and max_state_repeats to the logdir but put the random seed at the back
            logdir_split.insert(-1, str(10000))
            logdir_split.insert(-1, str(10000))
            logdir = "/".join(logdir_split)
            config.update({"logdir": logdir})

    else:

        puzzle = sys.argv[1]
        params = sys.argv[2]
        seed = int(sys.argv[3])
        logdir = sys.argv[4]
        max_env_steps = int(sys.argv[5])
        max_state_repeats = int(sys.argv[6])
        test = False

        # See configs.yaml for all options.
        config = embodied.Config(dreamerv3.configs["defaults"])
        # config = config.update(dreamerv3.configs['debug'])
        config = config.update(dreamerv3.configs["small"])
        config = config.update(
            {
                "logdir": logdir,
                "run.train_ratio": 512,
                "run.log_every": 1200,  # Seconds
                "batch_size": 16,
                "jax.prealloc": False,
                "encoder.mlp_keys": ".*",
                "decoder.mlp_keys": "^(?!reward$|is_terminal$).+",
                "encoder.cnn_keys": "$^",
                "decoder.cnn_keys": "$^",
                "seed": seed,
                "run.steps": 5e5,
                "run.max_env_steps": max_env_steps,
                "run.max_state_repeats": max_state_repeats,
                # 'jax.platform': 'cpu',
            }
        )
        # config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )
    if not test:
        config.save(logdir / "config.yaml")
    import crafter
    from embodied.envs import from_rlp, from_gym

    env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=puzzle,
        render_mode="rgb_array",
        params=params,
        obs_type="puzzle_state",
        max_state_repeats=max_state_repeats,
    )
    env.reset(seed=seed)
    env = from_rlp.FromRlp(
        env, obs_key="vector", max_episode_steps=max_env_steps
    )  # Or obs_key='vector', 'image'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    obs_space = env.obs_space
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )

    if test:
        args = args.update(
            {"from_checkpoint": os.path.join(config.logdir, "checkpoint.ckpt")}
        )
        embodied.run.eval_only(agent, env, logger, args, nr_eval_episodes=1000)
    else:
        embodied.run.train(agent, env, replay, logger, args)


if __name__ == "__main__":
    main()
