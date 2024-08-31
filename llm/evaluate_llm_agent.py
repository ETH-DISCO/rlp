import os
import pygame
import gymnasium as gym

import rlp

from llm.llm_api_agent.gemini_agent import GeminiPuzzlesAgent
from llm.llm_api_agent.chatgpt_agent import ChatGPTPuzzlesAgent

if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument(
        "-me",
        "--max_episode_steps",
        type=int,
        help="max steps in single episode",
        default=10000,
    )
    parser.add_argument(
        "-mt", "--model_type", type=str, help="model type", default="gemini-1.5-flash"
    )
    args = parser.parse_args()

    render_mode = "human" if not args.headless else "rgb_array"

    eval_env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=args.puzzle,
        render_mode=render_mode,
        params=args.arg,
        obs_type="rgb",
        max_episode_steps=args.max_episode_steps,
        max_state_repeats=5,
        window_width=800,
        window_height=800,
    )

    if "gemini" in args.model_type:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        agent = GeminiPuzzlesAgent(
            args.puzzle,
            "puzzle_state",
            args.arg,
            api_key=gemini_api_key,
            model=args.model_type,
        )
    elif "gpt" in args.model_type:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        agent = ChatGPTPuzzlesAgent(
            args.puzzle,
            "puzzle_state",
            args.arg,
            api_key=openai_api_key,
            model=args.model_type,
        )
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    image_obs, info = eval_env.reset()
    discrete_obs = str(info)

    nr_steps = 0

    while True:
        action_masks = eval_env.unwrapped.action_masks()
        action = agent.get_action(image_obs, discrete_obs, action_masks)
        image_obs, reward, done, truncated, info = eval_env.step(action)
        nr_steps += 1
        discrete_obs = str(info)
        eval_env.render()
        if truncated:
            print(f"Episode truncated after {nr_steps} steps due to state repeats")
            agent.reset()
            break
        if done:
            result = "Success" if reward > 0 else "Failure"
            print(f"Episode finished after {nr_steps} steps with result: {result}")
            agent.reset()
            break

    pygame.quit()
