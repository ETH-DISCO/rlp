import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # nopep8

from gymnasium.envs.registration import register
from rlp import constants, specific_api, api, puzzle, envs


def register_a_puzzle(internal_name: str, env_name: str):
    """Registers a specific puzzle as a gymnasium environment.

    Args:
        internal_name: The internally used name for the puzzle, see its C source file name.
        env_name: The camel case name for the puzzle.
    """
    register(
        id="rlp/" + env_name + "-v0",
        entry_point="rlp.envs:PuzzleEnv",
        kwargs={"puzzle": internal_name},
    )


supported_puzzles: dict[str, str] = {
    "blackbox": "BlackBox",
    "bridges": "Bridges",
    "cube": "Cube",
    "dominosa": "Dominosa",
    "fifteen": "Fifteen",
    "filling": "Filling",
    "flip": "Flip",
    "flood": "Flood",
    "galaxies": "Galaxies",
    "guess": "Guess",
    "inertia": "Inertia",
    "keen": "Keen",
    "lightup": "LightUp",
    "loopy": "Loopy",
    "magnets": "Magnets",
    "map": "Map",
    "mines": "Mines",
    "mosaic": "Mosaic",
    "net": "Net",
    "netslide": "Netslide",
    "palisade": "Palisade",
    "pattern": "Pattern",
    "pearl": "Pearl",
    "pegs": "Pegs",
    "range": "Range",
    "rect": "Rectangles",
    "samegame": "SameGame",
    "signpost": "Signpost",
    "singles": "Singles",
    "sixteen": "Sixteen",
    "slant": "Slant",
    "solo": "Solo",
    "tents": "Tents",
    "towers": "Towers",
    "tracks": "Tracks",
    "twiddle": "Twiddle",
    "undead": "Undead",
    "unruly": "Unruly",
    "unequal": "Unequal",
    "untangle": "Untangle",
}

# Register envs
register(
    id="rlp/Puzzle-v0",
    entry_point="rlp.envs:PuzzleEnv",
)

for internal_name, env_name in supported_puzzles.items():
    register_a_puzzle(internal_name, env_name)
