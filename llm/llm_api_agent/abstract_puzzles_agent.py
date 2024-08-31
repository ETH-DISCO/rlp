from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import os
import pygame
import string

import rlp


class AbstractPuzzlesAgent(ABC):
    def __init__(
        self,
        puzzle,
        obs_type,
        params,
        short_description=False,
        print_response=False,
        **kwargs,
    ):
        """
        Initializes the AbstractPuzzlesAgent.

        Args:
            puzzle: The name of the puzzle to solve.
            obs_type: The type of observation the puzzle uses.
            params: The parameters supplied to the puzzle.
            short_description: Whether to use a short description of the puzzle.
            print_response: Whether to print the API response.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.puzzle = puzzle
        self.obs_type = obs_type
        self.params = params
        self.use_short_description = short_description
        self.last_move = None
        self.print_response = print_response

        action_keys = rlp.puzzle.api.specific.get_action_keys(self.puzzle, False)
        self.key_names = [pygame.key.name(key) for key in action_keys]
        self.key_name_to_action = {
            name.translate(str.maketrans("", "", string.punctuation)): i
            for i, name in enumerate(self.key_names)
        }

    def get_system_prompt(self) -> str:
        """
        Constructs the system prompt for the language model.

        Returns:
            The system prompt as a string.
        """
        intro = (
            "You are a logic puzzle expert. You will soon be given a logic puzzle to solve. Here is a description "
            "of the puzzle:\n"
        )
        # load puzzle description
        desc_type = "short" if self.use_short_description else "long"
        file_path = os.path.dirname(__file__)
        description_file = os.path.join(
            file_path, f"../puzzle_explanations/{desc_type}/{self.puzzle}.txt"
        )
        with open(description_file, "r") as f:
            puzzle_description = f.read()
            intro += puzzle_description
        intro += "You will have to play the game using the cursor keys. It will work as follows:\n"
        intro += "You will be given the current state of the puzzle and you will have to make a move. "
        intro += "You will then be shown the new state of the puzzle. This will continue until you solve the puzzle. "
        intro += "Try to solve the puzzle as efficiently as possible. "
        intro += "Try to first describe the current state of the puzzle, taking both the image and the discrete state "
        intro += "representation into account. "
        intro += "Then, think about the move you want to make. Think step by step and "
        intro += "ensure that it satisfies the rules. "
        intro += "Also, make sure to not generate any unnecessary text after you have decided on a move. "
        intro += "Good luck!"
        return intro

    @abstractmethod
    def format_image_observation(self, observation: gym.core.ObsType) -> any:
        """
        Formats the image observation for the language model.

        Args:
            observation: The image observation.

        Returns:
            The formatted image observation.
        """
        pass

    def extract_action(self, response: str) -> tuple[int, str]:
        """
        Extracts the action from the language model's response.

        Args:
            response: The language model's response.

        Returns:
            The index of the extracted action and the action string.
        """
        last_line = response.splitlines()[-1]  # Get the last line
        parts = last_line.split("I choose ")
        if len(parts) == 2:
            action_str = parts[1].strip()  # Remove leading/trailing whitespace
        else:
            raise ValueError(f"Could not extract action from response: {response}")
        action_str = action_str.translate(str.maketrans("", "", string.punctuation))

        return (
            self.key_name_to_action[action_str.strip().lower()],
            action_str.strip().lower(),
        )

    def get_action_prompt(self, action_mask: np.ndarray) -> str:
        """
        Constructs the action prompt for the language model.

        Args:
            action_mask: A binary mask indicating valid actions.

        Returns:
            The action prompt as a string.
        """
        valid_actions = [
            self.key_names[i] for i in range(len(action_mask)) if action_mask[i]
        ]
        prompt = "What is your next move? Choose from the following valid options: "
        prompt += ", ".join(valid_actions)
        prompt += ". Explain your reasoning, and on a final new line, format your response as 'I choose [action]'"
        return prompt

    @abstractmethod
    def get_api_response(self, text_prompt: str, formatted_image: any) -> str:
        """
        Gets the response from the language model API.

        Args:
            text_prompt: The text prompt for the language model.
            formatted_image: The formatted image observation.

        Returns:
            The language model's response as a string.
        """
        pass

    def get_prompt(
        self,
        image_observation: np.ndarray,
        text_observation: str,
        action_mask: np.ndarray,
    ) -> tuple[str, any]:
        """
        Constructs the full prompt for the language model.

        Args:
            image_observation: The image observation.
            text_observation: The text observation.
            action_mask: A binary mask indicating valid actions.

        Returns:
            A tuple containing the full prompt and the formatted image observation.
        """
        prompt = self.get_system_prompt()
        prompt += "Here is the current state of the puzzle as a string of the internal state representation:\n"
        prompt += text_observation
        prompt += "\n You are also given the current state of the puzzle as an image. "
        prompt += f"Your last move was: [{self.last_move}].\n "
        prompt += self.get_action_prompt(action_mask)
        formatted_obs = self.format_image_observation(image_observation)
        return prompt, formatted_obs

    def get_action(
        self,
        image_observation: np.ndarray,
        text_observation: str,
        action_mask: np.ndarray,
    ) -> int:
        """
        Gets the action from the language model, given the observations and action mask.

        Args:
            image_observation: The image observation.
            text_observation: The text observation.
            action_mask: A binary mask indicating valid actions.

        Returns:
            The chosen action.
        """
        prompt, formatted_obs = self.get_prompt(
            image_observation, text_observation, action_mask
        )
        response = self.get_api_response(prompt, formatted_obs)
        try:
            action, self.last_move = self.extract_action(response)
        except Exception as e:
            print(f"Error extracting action from response: {response}")
            print(e)
            action = 0
        return action

    def reset(self):
        """
        Resets the agent's state.
        """
        self.last_move = None
