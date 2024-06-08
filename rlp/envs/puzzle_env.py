from typing import Any, Mapping
import threading

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import numpy as np
import pygame

from rlp import puzzle as rp
from rlp.envs import observation_spaces as obs_spaces

FPS = 60


class PuzzleEnv(gym.Env, EzPickle):
    """
    Puzzle Reinforcement Learning Environment.
    A Gym wrapper around the logic puzzles in
    Simon Tatham's Portable Puzzle Collection.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
        "observation_types": ["rgb", "puzzle_state"],
    }

    def __init__(
        self,
        puzzle: str,
        render_mode: str = "human",
        obs_type: str = "rgb",
        window_width: int = 128,
        window_height: int = 128,
        allow_undo: bool = False,
        max_state_repeats: int = 200,
        include_cursor_in_state_info: bool = True,
        params: str | None = None,
        n_envs: int = 1,
    ):
        if obs_type.lower() == "rgb":
            obs_type = "rgb"
        elif obs_type not in self.metadata["observation_types"]:
            raise ValueError(
                f"Invalid observation type: {obs_type}. Expecting: rgb, puzzle_state."
            )
        self.obs_type = obs_type

        EzPickle.__init__(**locals())
        gym.Env.__init__(self)
        self.n_envs = n_envs

        # Puzzle parameters
        self.puzzle_name = puzzle
        self.params = params
        # Pygame window dimensions
        self.window_width = window_width
        self.window_height = window_height

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the keyboard key to press
        I.e. 0 corresponds to "arrow_up", 1 to "arrow_down" etc.
        """
        self.allow_undo = allow_undo
        self.action_keys = rp.api.specific.get_action_keys(
            self.puzzle_name, self.allow_undo
        )
        self._action_to_key = {i: key for i, key in enumerate(self.action_keys)}

        # Tracking the CTRL / SHIFT modifiers
        self.modifiers_down = np.zeros((2), dtype=bool)
        self.modifiers_value = 0

        self.action_space = spaces.Discrete(len(self._action_to_key))

        self.reward_from_game_status = {
            -1: -100,
            0: 0,
            1: +100,
        }
        self.state_dict: dict = dict()
        self.state_histogram: dict[int, int] = dict()
        self.current_state_hash: int = 0
        self.include_cursor_in_state_info = include_cursor_in_state_info
        self.max_state_repeats = max_state_repeats
        self.current_move_was_toward_solution = False
        self.current_move_was_cursor_move = False
        self.next_move_strings: list[str | None] | None = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        `self.clock` will be a clock that is used to ensure that the environment
        is rendered at the correct framerate in human-mode. It will remain
        `None` until human-mode is used for the first time.
        """
        self.clock: pygame.time.Clock | None = None
        self.ticks = 0

        self.puzzle = rp.Puzzle(
            self.puzzle_name,
            self.window_width,
            self.window_height,
            self.params,
            False if self.render_mode == "human" else True,
        )

        if self.obs_type == "rgb":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        0,
                        255,
                        (3, self.window_width, self.window_height),
                        dtype=np.uint8,
                    ),
                }
            )
        elif self.obs_type == "puzzle_state":
            self.observation_space = spaces.Dict(
                obs_spaces.get_observation_space(
                    self.puzzle_name,
                    self.puzzle.fe.contents.me.contents.states[0].state.contents,
                    (
                        None
                        if self.puzzle_name in rp.api.specific.ui_reset_never
                        else self.puzzle.fe.contents.me.contents.ui
                    ),
                )
            )

    def _get_obs(self) -> dict:
        if self.obs_type == "rgb":
            return {
                "pixels": np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.puzzle.surf)),
                    axes=(2, 1, 0),
                )
            }
        else:
            if not self._state_dict_up_to_date:
                self._update_state_dict()
            return obs_spaces.get_observation(
                self.puzzle_name,
                self.state_dict,
                (
                    None
                    if self.include_cursor_in_state_info
                    else rp.api.specific.get_cursor_coords(
                        self.puzzle_name, self.puzzle.fe.contents.me.contents
                    )
                ),
            )

    def _update_state_dict(self) -> None:
        self.state_dict = self.puzzle.get_puzzle_state(
            self.include_cursor_in_state_info
        )
        self._state_dict_up_to_date = True

    def _get_info(self) -> dict[str, Any]:
        if not self._state_dict_up_to_date:
            self._update_state_dict()
        self.current_state_hash = rp.api.make_hash(self.state_dict)  # type: ignore[assignment]
        if self.current_state_hash in self.state_histogram:
            current_occurances = self.state_histogram[self.current_state_hash]
        else:
            current_occurances = 0
        self.state_histogram.update({self.current_state_hash: current_occurances + 1})
        return {
            "puzzle_state": self.state_dict,
            "state_histogram": self.state_histogram,
            "current_state_repeats": current_occurances + 1,
            "current_move_was_toward_solution": self.current_move_was_toward_solution,
            "current_move_was_cursor_move": self.current_move_was_cursor_move,
        }

    def action_masks(self) -> np.ndarray:
        action_mask, self.next_move_strings = self.puzzle.valid_action_mask(
            self.action_keys, self.modifiers_value
        )
        return action_mask

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize a new puzzle
        self.puzzle.new_game(self.allow_undo)
        self.puzzle.set_cursor_active()
        self.ticks = 0

        self.modifiers_down[:2] = [False, False]
        self.modifiers_value = 0
        self._state_dict_up_to_date = False
        self.state_dict.clear()
        self.state_histogram.clear()
        self.current_state_hash = 0
        self.current_move_was_toward_solution = False
        self.current_move_was_cursor_move = False
        self.next_move_strings = None

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action to the keyboard input
        key = self._action_to_key[action]

        # this is only available for puzzles that support
        # a solver (which provides a sequence of moves that
        # solve the puzzle)
        self.current_move_was_toward_solution = False
        if key in rp.api.specific.CURSOR_MOVE_KEYS and self.puzzle.can_solve:
            self.current_move_was_cursor_move = True
        else:
            self.current_move_was_cursor_move = False

        # CTRL/SHIFT: Consider an action to be changing the status only,
        # but for the other keys, consider it as a key press

        if key == pygame.K_LCTRL or key == pygame.K_LSHIFT:
            modifier = 0 if key == pygame.K_LCTRL else 1
            self.modifiers_down[modifier] ^= 1
            self.modifiers_value = (
                pygame.KMOD_CTRL if self.modifiers_down[0] else 0
            ) + (pygame.KMOD_SHIFT if self.modifiers_down[1] else 0)
        else:
            # when action masking is applied, we already have the next move's string
            # if not, we need to get it for further processing
            if self.next_move_strings and action < len(self.next_move_strings):
                current_move_string = self.next_move_strings[action]
            else:
                _, move_string_list = self.puzzle.valid_action_mask(
                    [key], self.modifiers_value, action
                )
                current_move_string = move_string_list[0] if move_string_list else None
            try:
                self.current_move_string = current_move_string.decode("utf-8")
                if len(self.current_move_string) > 1:
                    self.current_move_was_toward_solution = (
                        self.puzzle.check_move_against_solution(
                            self.current_move_string
                        )
                    )
            except:
                self.current_move_string = ""

            self.puzzle.process_key(pygame.KEYDOWN, key, self.modifiers_value)
        self._state_dict_up_to_date = False

        # Status: {+1: completed, 0: ongoing, -1: failed/died/stuck}
        status = self.puzzle.game_status()
        reward = self.reward_from_game_status[status]

        observation = self._get_obs()
        info = self._get_info()

        # An episode is done iff the puzzle backend deems it
        # to be in a "solved" or "failed" state
        terminated = True if abs(status) == 1 else False

        truncated = False
        if self.state_histogram[self.current_state_hash] > self.max_state_repeats:
            truncated = True

        self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            pygame.event.pump()
            # Delays the next frame to keep it relatively accurate to the desired framerate
            self.clock.tick(self.metadata["render_fps"])
        self.puzzle.proceed_animation()

        if not self.render_mode == "human":  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.puzzle.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.puzzle is not None:
            self.puzzle.destroy()

    def get_state(self) -> Mapping[str, Any]:
        env_dict = self.__dict__.copy()
        state_dict: dict[str, Any] = {}
        state_dict["env_dict"] = {
            k: v for k, v in env_dict.items() if k not in ["puzzle"]
        }
        state_dict["puzzle"] = self.puzzle.serialise_state(
            "/tmp/rlp", threading.get_native_id()
        )
        return state_dict

    def set_state(self, state_dict: Mapping[str, Any]):
        self.puzzle.deserialise_state(state_dict["puzzle"])
        self.__dict__.update(state_dict["env_dict"])

        observation = self._get_obs()
        return_observation = {}
        return_observation["obs"] = spaces.utils.flatten(
            self.observation_space, observation
        )
        return_observation["action_mask"] = self.action_masks()
        return return_observation
