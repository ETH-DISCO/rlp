from typing import Callable, Sequence

import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, MultiDiscrete, Space
import numpy as np

from rlp import puzzle as rp
from rlp.specific_api import GameState, GameUi

# Helper functions
get_binary_obs = lambda x: 1 if x else 0


def split_int_to_bit_array(
    integer: int,
    bit_array,
    bit_selector_fn: Callable[[int], int] | None = None,
    bit_shift_fn: Callable[[int], int] | None = None,
):
    """Splits an integer into its component bits."""
    if bit_selector_fn is None:
        bit_selector_fn = lambda x: integer
    if bit_shift_fn is None:
        bit_shift_fn = lambda x: x
    for bit in range(len(bit_array)):
        bit_array[bit] = (bit_selector_fn(bit) >> bit_shift_fn(bit)) & 1


# Wrapper functions for puzzle-specific getters
def get_observation_space(
    puzzle: str, game_state: GameState, game_ui: GameUi | None = None
) -> dict:
    try:
        return get_observation_space_methods[puzzle](game_state)
    except TypeError:
        return get_observation_space_methods[puzzle](game_state, game_ui)
    except KeyError:
        raise KeyError(f"Cannot get a valid observation space for puzzle {puzzle}.")


def get_observation(
    puzzle: str, info: dict, cursor_pos: tuple[int, int] | None = None
) -> dict:
    try:
        ret = get_observation_methods[puzzle](info)
        if puzzle not in rp.api.specific.ui_reset_never:
            ret.update(
                {
                    "cursor_pos": np.asarray(
                        cursor_pos if cursor_pos else info["cursor_pos"], dtype=np.int32
                    )
                }
            )
        return ret
    except KeyError:
        raise KeyError(f"Cannot get a valid observation for puzzle {puzzle}.")


UINT_MAX = rp.api.c.c_uint(~0).value
INT_MAX = rp.api.c.c_uint(~0).value // 2
LONG_MAX = rp.api.c.c_ulong(~0).value // 2


# Puzzle-specific getter functions
def get_observation_space_blackbox(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h

    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 2, h + 2], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "minballs": Discrete(game_state.minballs, start=1),
        "maxballs": Discrete(game_state.maxballs, start=1),
        "nballs": Discrete(game_state.nballs, start=1),
        "nlasers": Discrete(game_state.nlasers, start=1),
        "grid": MultiDiscrete([w + h + 3] * ((w + 2) * (h + 2)), dtype=np.int32),
        "exits": Box(0, 4294967295, (game_state.nlasers,), dtype=np.uint32),
        "laserno": Discrete(game_state.nlasers, start=1),
        "nguesses": Discrete(w * h + 1),
        "nright": Discrete(game_state.maxballs + 1),
        "nwrong": Discrete(game_state.maxballs + 1),
        "nmissed": Discrete(game_state.maxballs + 1),
    }


def get_observation_blackbox(state: dict) -> dict:
    FLAG_CURSOR = 65536
    w = state["w"]
    h = state["h"]
    grid_arr = np.zeros(((w + 2) * (h + 2),), dtype=np.int32)
    exits_arr = np.zeros((state["nlasers"],), dtype=np.uint32)
    for i, e in enumerate(state["grid"]):
        square = e & ~FLAG_CURSOR
        if square == 8192:
            grid_arr[i] = 0
        elif square == 4096:
            grid_arr[i] = 1
        else:
            grid_arr[i] = square + 2
    for i, e in enumerate(state["exits"]):
        exits_arr[i] = e

    return {
        "w": w,
        "h": h,
        "minballs": state["minballs"],
        "maxballs": state["maxballs"],
        "nballs": state["nballs"],
        "nlasers": state["nlasers"],
        "grid": grid_arr,
        "exits": exits_arr,
        "laserno": state["laserno"],
        "nguesses": state["nguesses"],
        "nright": state["nright"],
        "nwrong": state["nwrong"],
        "nmissed": state["nmissed"],
    }


def get_observation_space_bridges(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    max_wh = max(w, h)
    return {
        "cursor_pos": MultiDiscrete(nvec=[w, h], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "max_bridges": Discrete(4, start=1),
        "allowloops": Discrete(2),
        "grid": Box(0, 8091, (w * h,), dtype=np.uint32),
        "n_islands": Discrete(w * h, start=1),
        "islands": Dict(
            spaces={
                "coords": Box(low=-1, high=w * h, shape=(w * h * 2,), dtype=np.int8),
                "target_line_count": Box(
                    low=0, high=w * h, shape=(w * h,), dtype=np.uint8
                ),
                "orthogonal_offsets": Box(
                    low=-max_wh + 1, high=max_wh - 1, shape=(w * h * 8,), dtype=np.int8
                ),
            }
        ),
        "lines": MultiDiscrete(nvec=[5] * (w * h)),
    }


def get_observation_bridges(state: dict) -> dict:
    w = state["w"]
    h = state["h"]
    coords = np.full((w * h * 2,), -1, dtype=np.int8)
    target_line_count = np.full((w * h,), 0, dtype=np.uint8)
    orthogonal_offsets = np.zeros((w * h * 8,), dtype=np.int8)
    for island in state["islands"]:
        index = island["y"] * w + island["x"]
        i2 = index * 2
        i8 = index * 8
        coords[i2] = island["y"]
        coords[i2 + 1] = island["x"]
        target_line_count[index] = island["count"]
        for i, adj in enumerate(island["adj"]["points"]):
            orthogonal_offsets[i8 + 2 * i] = adj["dx"] * adj["off"]
            orthogonal_offsets[i8 + 2 * i + 1] = adj["dy"] * adj["off"]

    return {
        "w": w,
        "h": h,
        "max_bridges": state["params"]["maxb"],
        "allowloops": get_binary_obs(state["allowloops"]),
        "grid": np.array(state["grid"], dtype=np.uint32),
        "n_islands": state["n_islands"],
        "islands": {
            "coords": coords,
            "target_line_count": target_line_count,
            "orthogonal_offsets": orthogonal_offsets,
        },
        "lines": np.array(state["lines"], dtype=np.uint8),
    }


CUBE_SOLIDS = {
    "MAXVERTICES": 20,
    "MAXFACES": 20,
    "MAXORDER": 4,
}


def get_observation_space_cube(game_state: GameState) -> dict[str, Space]:
    d1 = game_state.params.d1
    d2 = game_state.params.d2
    nsquares = (
        d1 * d2
        if game_state.solid.contents.order == 4
        else d1 * d1 + d2 * d2 + 4 * d1 * d2
    )
    nvertices = game_state.solid.contents.nvertices

    return {
        "order": Discrete(nsquares, start=1),
        "nsquares": Discrete(nsquares, start=1),
        "nvertices": Discrete(nvertices, start=1),
        "vertices": Box(-1, 1, (CUBE_SOLIDS["MAXVERTICES"] * 3,), np.float32),
        "nfaces": Discrete(nsquares, start=1),
        "faces": MultiDiscrete(
            [nvertices] * (CUBE_SOLIDS["MAXVERTICES"] * CUBE_SOLIDS["MAXORDER"]),
            np.int32,
        ),
        "normals": Box(-1, 1, (CUBE_SOLIDS["MAXFACES"] * 3,), np.float32),
        "shear": Box(-1, 1, dtype=np.float32),
        "border": Box(-1, 1, dtype=np.float32),
        "facecolours": MultiBinary(n=game_state.solid.contents.nfaces),
        "bluemask": MultiBinary(nsquares),
        "previous": Discrete(nsquares),
        "current": Discrete(nsquares),
        "angle": Box(-2 * np.pi, 2 * np.pi, dtype=np.float32),
        "sgkey": MultiDiscrete(nvec=[nsquares] * 2, dtype=np.uint32),
        "dgkey": MultiDiscrete(nvec=[nsquares] * 2, dtype=np.uint32),
        "spkey": MultiDiscrete(nvec=[nvertices] * 2, dtype=np.uint32),
        "dpkey": MultiDiscrete(nvec=[nvertices] * 2, dtype=np.uint32),
    }


def get_observation_cube(state: dict) -> dict:
    grid = state["grid"]
    nsquares = grid["nsquares"]

    bluemask = np.zeros((nsquares,), dtype=np.uint8)
    for square in range(nsquares):
        bluemask[square] = (state["bluemask"][square // 32] >> (square % 32)) & 1
    return {
        "order": state["solid"]["order"],
        "nsquares": nsquares,
        "nvertices": state["solid"]["nvertices"],
        "vertices": np.array(state["solid"]["vertices"], np.float32),
        "nfaces": state["solid"]["nfaces"],
        "faces": np.array(state["solid"]["faces"], np.int32),
        "normals": np.array(state["solid"]["normals"], np.float32),
        "shear": np.array([state["solid"]["shear"]], dtype=np.float32),
        "border": np.array([state["solid"]["border"]], dtype=np.float32),
        "facecolours": np.array(state["facecolours"]),
        "bluemask": bluemask,
        "previous": state["previous"],
        "current": state["current"],
        "angle": np.array([state["angle"]], dtype=np.float32),
        "sgkey": np.array(state["sgkey"], dtype=np.uint32),
        "dgkey": np.array(state["dgkey"], dtype=np.uint32),
        "spkey": np.array(state["spkey"], dtype=np.uint32),
        "dpkey": np.array(state["dpkey"], dtype=np.uint32),
    }


DOMINOSA_EDGES = {
    0: 0,  # EDGE_EMPTY
    256: 1,  # EDGE_L
    512: 2,  # EDGE_R
    1024: 3,  # EDGE_T
    2048: 4,  # EDGE_B
}


def get_observation_space_dominosa(game_state: GameState) -> dict[str, Space]:
    n = game_state.params.n
    w = game_state.w
    h = game_state.h
    wh = w * h
    maxwh = max(w, h)
    return {
        "cursor_pos": MultiDiscrete(nvec=[2 * maxwh, 2 * maxwh], dtype=np.int32),
        "n": Discrete(n, start=1),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "numbers": MultiDiscrete(nvec=[n + 1] * wh, dtype=np.uint16),
        "grid": MultiDiscrete(nvec=[wh + 1] * wh, dtype=np.uint16),
        "edges": MultiDiscrete([len(DOMINOSA_EDGES)] * wh, dtype=np.uint8),
    }


def get_observation_dominosa(state: dict) -> dict:
    w = state["w"]
    h = state["h"]
    edges = np.zeros((w * h,), dtype=np.uint8)
    for edge in state["edges"]:
        edges[edge] = DOMINOSA_EDGES[edge]
    return {
        "n": state["params"]["n"],
        "w": w,
        "h": h,
        "numbers": np.array(state["numbers"]["numbers"], dtype=np.uint16),
        "grid": np.array(state["grid"], dtype=np.uint16),
        "edges": edges,
    }


def get_observation_space_fifteen(game_state: GameState) -> dict[str, Space]:
    n = game_state.n
    w = game_state.w
    h = game_state.h
    return {
        "n": Discrete(n, start=1),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "tiles": MultiDiscrete([n] * n, dtype=np.uint16),
        "gap_pos": Discrete(n),
    }


def get_observation_fifteen(state: dict) -> dict:
    w = state["w"]
    h = state["h"]
    return {
        "n": state["n"],
        "w": w,
        "h": h,
        "tiles": np.array(state["tiles"], dtype=np.uint16),
        "gap_pos": state["gap_pos"],
    }


def get_observation_space_filling(game_state: GameState) -> dict[str, Space]:
    w = game_state.shared.contents.params.w
    h = game_state.shared.contents.params.h
    wh = w * h
    maxwh = max(w, h)
    return {
        "cursor_pos": MultiDiscrete(nvec=[maxwh, maxwh], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "board": MultiDiscrete([10] * wh, dtype=np.uint16),
        "clues": MultiDiscrete([10] * wh, dtype=np.uint16),
    }


def get_observation_filling(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "board": np.array(state["board"], dtype=np.uint16),
        "clues": np.array(state["shared"]["clues"], dtype=np.uint16),
    }


def get_observation_space_flip(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "grid": MultiDiscrete([4] * (wh)),
        "matrix": MultiBinary(wh * wh),
    }


def get_observation_flip(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "grid": np.array(state["grid"], dtype=np.uint8),
        "matrix": np.array(state["matrix"]["matrix"], dtype=np.uint8),
    }


def get_observation_space_flood(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "colours": Discrete(game_state.colours, start=1),
        "grid": MultiDiscrete([game_state.colours] * w * h, dtype=np.uint16),
    }


def get_observation_flood(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "colours": state["colours"],
        "grid": np.array(state["grid"], dtype=np.uint16),
    }


GALAXIES_FLAGS = {
    0: 0,  # EMPTY
    1: 1,  # F_DOT
    2: 2,  # F_EDGE_SET
    4: 3,  # F_TILE_ASSOC
    8: 4,  # F_DOT_BLACK
    16: 5,  # F_MARK
    32: 6,  # F_REACHABLE
    64: 7,  # F_SCRATCH
    128: 8,  # F_MULTIPLE
    256: 9,  # F_DOT_HOLD
    512: 10,  # F_GOOD
}


def get_observation_space_galaxies(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    sx = game_state.sx
    sy = game_state.sy
    maxsxsy = max(sx, sy)
    sxsy = sx * sy
    return {
        "cursor_pos": MultiDiscrete(nvec=[2 * w, 2 * h], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "sx": Discrete(sx, start=1),
        "sy": Discrete(sy, start=1),
        "grid": Dict(
            spaces={
                "coords": MultiDiscrete(nvec=[maxsxsy] * sxsy * 2, dtype=np.uint16),
                "type": MultiDiscrete(nvec=[3] * sxsy, dtype=np.uint16),
                "flags": MultiDiscrete(
                    nvec=[len(GALAXIES_FLAGS)] * sxsy, dtype=np.uint32
                ),
                "dot_coords": Box(-1, max(2 * w, 2 * h), (sxsy * 2,), dtype=np.int32),
            }
        ),
        "ndots": Discrete(wh, start=1),
        "dots_indices": Box(low=-1, high=2 * sxsy, shape=(wh,), dtype=np.int32),
    }


def get_observation_galaxies(state: dict) -> dict:
    sxsy = state["sx"] * state["sy"]
    wh = state["w"] * state["h"]
    grid_coords = np.zeros((sxsy * 2,), dtype=np.uint16)
    grid_type = np.zeros((sxsy,), dtype=np.uint16)
    grid_flags = np.zeros((sxsy,), dtype=np.uint32)
    grid_dot_coords = np.zeros((sxsy * 2,), dtype=np.int32)
    for i, space in enumerate(state["grid"]):
        grid_coords[i * 2] = space["x"]
        grid_coords[i * 2 + 1] = space["y"]
        grid_type[i] = space["type"]
        grid_flags[i] = GALAXIES_FLAGS[space["flags"]]
        grid_dot_coords[i * 2] = space["dotx"]
        grid_dot_coords[i * 2 + 1] = space["doty"]
    dots_indices = np.full((wh,), -1, dtype=np.int32)
    dots_indices[: state["ndots"]] = state["dots_indices"]

    return {
        "w": state["w"],
        "h": state["h"],
        "sx": state["sx"],
        "sy": state["sy"],
        "grid": {
            "coords": grid_coords,
            "type": grid_type,
            "flags": grid_flags,
            "dot_coords": grid_dot_coords,
        },
        "ndots": state["ndots"],
        "dots_indices": dots_indices,
    }


def get_observation_space_guess(game_state: GameState) -> dict[str, Space]:
    npegs = game_state.params.npegs
    ncolours = game_state.params.ncolours
    nguesses = game_state.params.nguesses
    pegs_grid_size = nguesses * npegs
    return {
        "cursor_pos": MultiDiscrete(nvec=[npegs + 1, ncolours], dtype=np.int32),
        "npegs": Discrete(npegs, start=1),
        "ncolours": Discrete(ncolours, start=1),
        "nguesses": Discrete(nguesses, start=1),
        "allow_blank": Discrete(2),
        "allow_multiple": Discrete(2),
        "guesses": Dict(
            spaces={
                "pegs": Box(0, ncolours, (pegs_grid_size,), dtype=np.uint16),
                "feedback": Box(0, 2, (pegs_grid_size,), dtype=np.uint16),
            }
        ),
        "current_guess": MultiDiscrete(nvec=[ncolours + 1] * npegs, dtype=np.uint16),
        "holds": MultiBinary(npegs),
        "next_go": Discrete(nguesses + 1, start=0),
    }


def get_observation_guess(state: dict) -> dict:
    npegs = state["params"]["npegs"]
    ncolours = state["params"]["ncolours"]
    nguesses = state["params"]["nguesses"]
    guesses_pegs = np.zeros((npegs * nguesses,), dtype=np.uint16)
    guesses_feedback = np.zeros((npegs * nguesses,), dtype=np.uint16)
    holds = np.zeros(npegs, dtype=np.int32)
    for i, guess in enumerate(state["guesses"]):
        guesses_pegs[i * npegs : (i + 1) * npegs] = guess["pegs"]
        guesses_feedback[i * npegs : (i + 1) * npegs] = guess["feedback"]
    for i, hold in enumerate(state["holds"]):
        holds[i] = get_binary_obs(hold)
    return {
        "npegs": npegs,
        "ncolours": ncolours,
        "nguesses": nguesses,
        "allow_blank": get_binary_obs(state["params"]["allow_blank"]),
        "allow_multiple": get_binary_obs(state["params"]["allow_multiple"]),
        "guesses": {
            "pegs": guesses_pegs,
            "feedback": guesses_feedback,
        },
        "current_guess": np.array(state["current_guess"]["pegs"], dtype=np.uint16),
        "holds": holds,
        "next_go": state["next_go"],
    }


INERTIA_GRID = {
    98: 0,  # BLANK
    103: 1,  # GEM
    109: 2,  # MINE
    115: 3,  # STOP
    119: 4,  # WALL
    83: 5,  # START
}


def get_observation_space_inertia(game_state: GameState) -> dict[str, Space]:
    w = game_state.params.w
    h = game_state.params.h
    wh = w * h
    return {
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "px": Discrete(w + 1),
        "py": Discrete(h + 1),
        "gems": Discrete(wh, start=1),
        "grid": MultiDiscrete(nvec=[len(INERTIA_GRID) + 1] * wh, dtype=np.uint16),
        "distance_moved": Discrete(max(w, h) * 2),
    }


def get_observation_inertia(state: dict) -> dict:
    w = state["params"]["w"]
    h = state["params"]["h"]
    grid_array = np.zeros((w * h,), dtype=np.uint16)
    for i, square in enumerate(state["grid"]):
        grid_array[i] = INERTIA_GRID[square]
    return {
        "w": w,
        "h": h,
        "px": state["px"],
        "py": state["py"],
        "gems": state["gems"],
        "grid": grid_array,
        "distance_moved": state["distance_moved"],
    }


KEEN_CLUES = {
    0: 0,  # C_ADD
    536870912: 1,  # C_MUL
    1073741824: 2,  # C_SUB
    1610612736: 3,  # C_DIV
}

KEEN_MAX_CLUE_VALUES = {
    3: 18,  # 3 * 2 * 3
    4: 144,  # 4 * 3 * 4 * 3
    5: 2000,  # 5 * 4 * 5 * 4 * 5
    6: 27000,  # 6 * 5 * 6 * 5 * 6 * 5
    7: 74088,  # 7 * 6 * 7 * 6 * 7 * 6
    8: 175616,  # 8 * 7 * 8 * 7 * 8 * 7
    9: 373248,  # 9 * 8 * 9 * 8 * 9 * 8
}


def get_observation_space_keen(game_state: GameState) -> dict[str, Space]:
    w = game_state.par.w
    w2 = w * w
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, w + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "multiplication_only": Discrete(2),
        "grid": MultiDiscrete([10] * w2, np.uint8),
        "dsf_is_root": MultiBinary(w2),
        "dsf_parent": MultiDiscrete([w2] * w2, dtype=np.int32),
        "dsf_size": MultiDiscrete([7] * w2, dtype=np.int32),
        "dsf_opposite_parent": MultiBinary(w2),
        "clue_ops": MultiDiscrete([4] * w2, dtype=np.uint8),
        "clue_values": Box(0, KEEN_MAX_CLUE_VALUES[w], (w2,), dtype=np.int64),
        "pencil": MultiBinary(w * w2),
    }


def get_observation_keen(state: dict) -> dict:
    CMASK = 1610612736
    w = state["par"]["w"]
    w2 = w * w
    clue_ops = np.zeros((w2), dtype=np.uint8)
    clue_values = np.zeros((w2), dtype=np.int64)
    for i, clue in enumerate(state["clues"]["clues"]):
        clue_ops[i] = KEEN_CLUES[clue & CMASK]
        clue_values[i] = clue & ~CMASK

    pencil = np.zeros((w * w2,), np.uint8)
    start = 0
    for square in range(w2):
        split_int_to_bit_array(
            state["pencil"][square],
            pencil[start : start + w],
            bit_shift_fn=lambda x: x + 1,
        )
        start += w

    return {
        "w": w,
        "multiplication_only": get_binary_obs(state["par"]["multiplication_only"]),
        "grid": np.array(state["grid"], dtype=np.uint8),
        "dsf_is_root": np.array(state["clues"]["dsf"]["is_root"]),
        "dsf_parent": np.array(state["clues"]["dsf"]["parent"], dtype=np.int32),
        "dsf_size": np.array(state["clues"]["dsf"]["size"], dtype=np.int32),
        "dsf_opposite_parent": np.array(state["clues"]["dsf"]["opposite_parent"]),
        "clue_ops": clue_ops,
        "clue_values": clue_values,
        "pencil": pencil,
    }


def get_observation_space_lightup(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "nlights": Discrete(wh + 1),
        "times_lit": MultiDiscrete([w + h] * wh, dtype=np.uint16),
        "black_squares": MultiBinary(wh),
        "lights_set_forbidden": MultiBinary(wh),
        "light": MultiBinary(wh),
        "numbers": MultiDiscrete([5] * wh),
    }


def get_observation_lightup(state: dict) -> dict:
    w = state["w"]
    h = state["h"]
    wh = w * h
    lights = state["lights"]
    times_lit = np.zeros((wh,), np.uint16)
    black_squares = np.zeros((wh,), np.uint16)
    lights_set_forbidden = np.zeros((wh,), np.uint16)
    light = np.zeros((wh,), np.uint16)
    numbers = np.zeros((wh,), np.uint16)

    for i, flag in enumerate(state["flags"]):
        if flag & 1:  # Black square
            numbers[i] = lights[i]
            black_squares[i] = 1
        else:
            times_lit[i] = lights[i]
            if flag & 8:  # lights not allowed in this square
                lights_set_forbidden[i] = 1
            if flag & 16:  # this square is a light source
                light[i] = 1

    return {
        "w": w,
        "h": h,
        "nlights": state["nlights"],
        "times_lit": times_lit,
        "black_squares": black_squares,
        "lights_set_forbidden": lights_set_forbidden,
        "light": light,
        "numbers": numbers,
    }


def get_observation_space_loopy(game_state: GameState) -> dict[str, Space]:
    nfaces = game_state.game_grid.contents.num_faces
    nlines = game_state.game_grid.contents.num_edges

    return {
        "cursor_pos": MultiDiscrete(
            nvec=[
                game_state.game_grid.contents.highest_x,
                game_state.game_grid.contents.highest_y,
            ],
            dtype=np.int32,
        ),
        "num_faces": Discrete(nfaces + 1),
        # getting the max number of dots is non-trivial, so we'll give it an upper bound of 2x the number of edges
        "game_grid": MultiBinary([nfaces, nlines, nlines * 2]),
        "clues": Box(low=-1, high=127, shape=(nfaces,), dtype=np.int8),
        "lines": Box(0, 2, (nlines,), dtype=np.uint8),
        "line_errors": MultiBinary(nlines),
    }


def get_observation_loopy(state: dict) -> dict:
    nlines = state["game_grid"]["num_edges"]
    nfaces = state["game_grid"]["num_faces"]

    game_grid = np.zeros((nfaces, nlines, nlines * 2), dtype=np.uint8)
    for edge_index, edge in enumerate(state["game_grid"]["edges"]):
        dot1_index = edge["dot1_index"]
        dot2_index = edge["dot2_index"]
        face1_index = edge["face1_index"]
        face2_index = edge["face2_index"]
        if face1_index:
            game_grid[face1_index, edge_index, dot1_index] = 1
            game_grid[face1_index, edge_index, dot2_index] = 1
        if face2_index:
            game_grid[face2_index, edge_index, dot1_index] = 1
            game_grid[face2_index, edge_index, dot2_index] = 1

    return {
        "num_faces": nfaces,
        "game_grid": game_grid,
        "clues": np.array(state["clues"], dtype=np.int8),
        "lines": np.array(state["lines"], dtype=np.uint8),
        "line_errors": np.array(state["line_errors"], dtype=np.uint8),
    }


def get_observation_space_magnets(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = game_state.wh
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "wh": Discrete(wh, start=1),
        "numbered": Discrete(2),
        "grid": MultiDiscrete(nvec=[3] * wh, dtype=np.uint8),
        "flags": MultiDiscrete(nvec=[33] * wh, dtype=np.uint8),
        "dominoes": MultiDiscrete(nvec=[wh] * wh, dtype=np.int16),
        "rowcount": Box(low=-1, high=3, shape=(3 * h,), dtype=np.int8),
        "colcount": Box(low=-1, high=3, shape=(3 * w,), dtype=np.int8),
        "counts_done": MultiBinary(2 * (w + h)),
    }


def get_observation_magnets(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "wh": state["wh"],
        "numbered": get_binary_obs(state["wh"]),
        "grid": np.array(state["grid"], dtype=np.uint8),
        "flags": np.array(state["flags"], dtype=np.uint8),
        "dominoes": np.array(state["common"]["dominoes"], dtype=np.int16),
        "rowcount": np.array(state["common"]["rowcount"], dtype=np.int8),
        "colcount": np.array(state["common"]["colcount"], dtype=np.int8),
        "counts_done": np.array(state["counts_done"], dtype=np.int8),
    }


def get_observation_space_map(game_state: GameState) -> dict[str, Space]:
    w = game_state.p.w
    h = game_state.p.h
    n = game_state.p.n
    wh = w * h
    n2 = n * n
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(n, start=1),
        "drag_colour": Discrete(6, start=-2),
        "map": Dict(
            spaces={
                "map": Box(0, n - 1, (4 * wh,), np.uint16),
                "graph": Box(0, n2 - 1, (n2,), np.uint16),
                "immutable": MultiBinary(n),
                "edgex": Box(0, n2 - 1, (n2,), np.uint16),
                "regionx": Box(0, n2 - 1, (n,), np.uint16),
                "regiony": Box(0, n2 - 1, (n,), np.uint16),
            }
        ),
    }


def get_observation_map(state: dict) -> dict:
    n = state["p"]["n"]
    edgex = np.zeros((n * n,), np.uint16)
    edgex[: state["map"]["ngraph"]] = state["map"]["edgex"]
    return {
        "w": state["p"]["w"],
        "h": state["p"]["h"],
        "n": n,
        "drag_colour": state["drag_colour"],
        "map": {
            "map": np.array(state["map"]["map"], np.uint16),
            "graph": np.array(state["map"]["graph"], np.uint16),
            "immutable": np.array(state["map"]["immutable"], np.uint16),
            "edgex": edgex,
            "regionx": np.array(state["map"]["regionx"], np.uint16),
            "regiony": np.array(state["map"]["regiony"], np.uint16),
        },
    }


def get_observation_space_mines(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    n = game_state.n
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(n + 1, start=0),
        "dead": Discrete(2),
        "unique": Discrete(2),
        "mines": MultiBinary(wh),
        "grid": Box(-3, 8, (wh,), dtype=np.int8),
    }


def get_observation_mines(state: dict) -> dict:
    w = state["w"]
    h = state["h"]
    wh = w * h
    if len(state["layout"]["mines"]) > 0:
        mines = np.array(state["layout"]["mines"], np.int8)
    else:
        mines = np.zeros((wh,), np.int8)

    return {
        "w": state["w"],
        "h": state["h"],
        "n": state["n"],
        "dead": get_binary_obs(state["dead"]),
        "unique": get_binary_obs(state["layout"]["unique"]),
        "mines": mines,
        "grid": np.array(state["grid"], np.int8),
    }


MOSAIC_CELL_STATES = {
    0: 0,  # STATE_UNMARKED
    1: 1,  # STATE_MARKED
    2: 2,  # STATE_BLANK
    4: 3,  # STATE_SOLVED
    8: 4,  # STATE_ERROR / STATE_UNMARKED_ERROR
    9: 5,  # STATE_MARKED_ERROR
    10: 6,  # STATE_BLANK_ERROR
    6: 7,  # STATE_BLANK_SOLVED
    5: 8,  # STATE_MARKED_SOLVED
    3: 9,  # STATE_OK_NUM
}


def get_observation_space_mosaic(game_state: GameState) -> dict[str, Space]:
    w = game_state.width
    h = game_state.height
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "width": Discrete(w, start=1),
        "height": Discrete(h, start=1),
        "not_completed_clues": Discrete(wh + 1, start=1),
        "cells_contents": MultiDiscrete([len(MOSAIC_CELL_STATES)] * wh, np.int8),
        "clues": Box(-1, 9, (wh,), np.int8),
    }


def get_observation_mosaic(state: dict) -> dict:
    w = state["width"]
    h = state["height"]
    wh = w * h
    clues = np.full((wh,), -1, np.int8)
    cells_contents = np.zeros((wh,), np.int8)
    for i, cell_state in enumerate(state["cells_contents"]):
        cells_contents[i] = MOSAIC_CELL_STATES[cell_state]
    for i in range(wh):
        clues[i] = state["board"]["actual_board"][i]["clue"]
    return {
        "width": w,
        "height": h,
        "not_completed_clues": state["not_completed_clues"],
        "cells_contents": cells_contents,
        "clues": clues,
    }


def get_observation_space_net(game_state: GameState) -> dict[str, Space]:
    w = game_state.width
    h = game_state.height
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "width": Discrete(w, start=1),
        "height": Discrete(h, start=1),
        "wrapping": Discrete(2),
        "last_rotate_x": Discrete(w + 1, start=-1),
        "last_rotate_y": Discrete(h + 1, start=-1),
        "last_rotate_dir": Discrete(4, start=-1),
        "tiles": MultiDiscrete([31] * wh, np.uint16),
        "barriers": MultiDiscrete([13] * wh, np.uint16),
    }


def get_observation_net(state: dict) -> dict:
    return {
        "width": state["width"],
        "height": state["height"],
        "wrapping": get_binary_obs(state["wrapping"]),
        "last_rotate_x": state["last_rotate_x"],
        "last_rotate_y": state["last_rotate_y"],
        "last_rotate_dir": state["last_rotate_dir"],
        "tiles": np.array(state["tiles"], np.uint16),
        "barriers": np.array(state["imm"]["barriers"], np.uint16),
    }


def get_observation_space_netslide(game_state: GameState) -> dict[str, Space]:
    w = game_state.width
    h = game_state.height
    wh = w * h
    return {
        "cursor_pos": Box(-1, max(w, h), (2,), dtype=np.int32),
        "width": Discrete(w, start=1),
        "height": Discrete(h, start=1),
        "wrapping": Discrete(2),
        "move_count": Box(0, INT_MAX, dtype=np.int32),
        "movetarget": Box(0, INT_MAX, dtype=np.int32),
        "last_move_row": Discrete(h + 1, start=-1),
        "last_move_col": Discrete(w + 1, start=-1),
        "last_move_dir": Discrete(4, start=-1),
        "tiles": Box(0, 31, (wh,), np.uint16),
        "barriers": Box(0, 256, (wh,), np.uint16),
    }


def get_observation_netslide(state: dict) -> dict:
    return {
        "width": state["width"],
        "height": state["height"],
        "wrapping": get_binary_obs(state["wrapping"]),
        "move_count": np.array([state["move_count"]], dtype=np.int32),
        "movetarget": np.array([state["movetarget"]], dtype=np.int32),
        "last_move_row": state["last_move_row"],
        "last_move_col": state["last_move_col"],
        "last_move_dir": state["last_move_dir"],
        "tiles": np.array(state["tiles"], np.uint16),
        "barriers": np.array(state["barriers"], np.uint16),
    }


def get_observation_space_palisade(game_state: GameState) -> dict[str, Space]:
    w = game_state.shared.contents.params.w
    h = game_state.shared.contents.params.h
    k = game_state.shared.contents.params.k
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "k": Discrete(k, start=1),
        "clues": Box(-1, 4, (wh,), np.int8),
        "borders": MultiDiscrete([16] * wh, np.int8),
    }


def get_observation_palisade(state: dict) -> dict:
    w = state["shared"]["params"]["w"]
    h = state["shared"]["params"]["h"]
    k = state["shared"]["params"]["k"]
    return {
        "w": w,
        "h": h,
        "k": k,
        "clues": np.array(state["shared"]["clues"], np.int8),
        "borders": np.array(state["borders"], np.int8),
    }


def get_observation_space_pattern(game_state: GameState) -> dict[str, Space]:
    w = game_state.common.contents.w
    h = game_state.common.contents.h
    wh = w * h
    wph = w + h
    maxwh = max(w, h)
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "rowsize": Discrete(maxwh, start=1),
        "rowlen": MultiDiscrete([maxwh + 1] * wph, np.uint16),
        "rowdata": MultiDiscrete([maxwh + 1] * (maxwh * wph), np.uint16),
        "immutable": MultiDiscrete([2] * wh, np.uint8),
        "grid": MultiDiscrete([3] * wh, np.uint8),
    }


def get_observation_pattern(state: dict) -> dict:
    w = state["common"]["w"]
    h = state["common"]["h"]
    wh = w * h
    maxwh = max(w, h)
    rowdata = np.zeros((maxwh * (w + h),), np.uint16)
    rowlen = np.array(state["common"]["rowlen"], np.uint16)
    start = cum_lengths = 0
    for length in rowlen:
        rowdata[start : start + length] = state["common"]["rowdata"][
            cum_lengths : cum_lengths + length
        ]
        start += maxwh
        cum_lengths += length

    return {
        "w": w,
        "h": h,
        "rowsize": state["common"]["rowsize"],
        "rowlen": rowlen,
        "rowdata": rowdata,
        "immutable": np.array(state["common"]["immutable"][:wh], np.uint8),
        "grid": np.array(state["grid"], np.uint8),
    }


def get_observation_space_pearl(game_state: GameState) -> dict[str, Space]:
    w = game_state.shared.contents.w
    h = game_state.shared.contents.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "sz": Discrete(wh, start=1),
        "clues": MultiDiscrete([3] * wh, np.uint8),
        "lines": Box(0, 15, (wh,), np.uint8),
        "errors": Box(0, 31, (wh,), np.uint8),
        "marks": Box(0, 15, (wh,), np.uint8),
        "ndragcoords": Discrete(wh + 2, start=-1),
        "dragcoords": MultiDiscrete([wh] * wh, np.uint16),
    }


def get_observation_pearl(state: dict) -> dict:
    w = state["shared"]["w"]
    h = state["shared"]["h"]
    wh = w * h
    ndragcoords = state["ndragcoords"]
    dragcoords = np.zeros((wh,), np.uint16)
    if ndragcoords > 0:
        dragcoords[:ndragcoords] = state["dragcoords"]

    return {
        "w": w,
        "h": h,
        "sz": state["shared"]["sz"],
        "clues": np.array(state["shared"]["clues"], np.uint8),
        "lines": np.array(state["lines"], np.uint8),
        "errors": np.array(state["errors"], np.uint8),
        "marks": np.array(state["marks"], np.uint8),
        "ndragcoords": ndragcoords,
        "dragcoords": dragcoords,
    }


def get_observation_space_pegs(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "grid": MultiDiscrete([3] * wh, dtype=np.uint8),
        "cur_jumping": Discrete(2),
    }


def get_observation_pegs(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "grid": np.array(state["grid"], np.uint8),
        "cur_jumping": get_binary_obs(state["cur_jumping"]),
    }


def get_observation_space_range(game_state: GameState) -> dict[str, Space]:
    w = game_state.params.w
    h = game_state.params.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "grid": Box(-2, 127, (wh,), np.int8),
    }


def get_observation_range(state: dict) -> dict:
    w = state["params"]["w"]
    h = state["params"]["h"]
    wh = w * h
    return {
        "w": w,
        "h": h,
        "grid": np.array(state["grid"], np.int8),
    }


def get_observation_space_rect(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "grid": Box(0, 127, (wh,), np.int8),
        "vedge": MultiDiscrete([2] * wh, np.int8),
        "hedge": MultiDiscrete([2] * wh, np.int8),
        "cur_dragging": Discrete(2),
        "drag_topleft": Box(-1, max(w, h), (2,), np.int16),
        "drag_bottomright": Box(-1, max(w, h), (2,), np.int16),
    }


def get_observation_rect(state: dict) -> dict:
    w = state["w"]
    h = state["h"]
    wh = w * h
    return {
        "w": w,
        "h": h,
        "grid": np.array(state["grid"], np.int8),
        "vedge": np.array(state["vedge"], np.int8),
        "hedge": np.array(state["hedge"], np.int8),
        "cur_dragging": get_binary_obs(state["cur_dragging"]),
        "drag_topleft": np.array([state["x1"], state["y1"]], np.int16),
        "drag_bottomright": np.array([state["x2"], state["y2"]], np.int16),
    }


def get_observation_space_samegame(game_state: GameState) -> dict[str, Space]:
    w = game_state.params.w
    h = game_state.params.h
    wh = w * h
    ncols = game_state.params.ncols
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(wh, start=1),
        "ncols": Discrete(ncols, start=1),
        "scoresub": Discrete(2, start=1),
        "score": Box(0, INT_MAX, dtype=np.uint32),
        "tiles": Box(0, ncols, (wh,), np.uint8),
        "nselected": Discrete(wh + 1),
        "selected_tiles": Box(0, 256, (wh,), np.uint16),
    }


def get_observation_samegame(state: dict) -> dict:
    w = state["params"]["w"]
    h = state["params"]["h"]
    wh = w * h
    return {
        "w": w,
        "h": h,
        "n": state["n"],
        "ncols": state["params"]["ncols"],
        "scoresub": state["params"]["scoresub"],
        "score": np.array([state["score"]], np.uint32),
        "tiles": np.array(state["tiles"], np.uint8),
        "nselected": state["nselected"],
        "selected_tiles": np.array(state["selected_tiles"], np.uint16),
    }


def get_observation_space_signpost(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(wh, start=1),
        "dirs": MultiDiscrete([8] * wh, np.uint8),
        "nums": Box(0, UINT_MAX, (wh,), np.uint32),
        "flags": MultiDiscrete([4] * wh, np.uint8),
        "next": Box(-1, wh, (wh,), np.int16),
        "prev": Box(-1, wh, (wh,), np.int16),
        "dsf_is_root": MultiBinary(wh),
        "dsf_parent": MultiDiscrete([wh] * wh, dtype=np.int32),
        "dsf_size": MultiDiscrete([7] * wh, dtype=np.int32),
        "dsf_opposite_parent": MultiBinary(wh),
        "numsi": Box(-1, wh, (wh + 1,), np.int16),
        "dragging": Discrete(2),
        "drag_is_from": Discrete(2),
        "drag_start": MultiDiscrete([w + 1, h + 1], np.int16),
    }


def get_observation_signpost(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "n": state["n"],
        "dirs": np.array(state["dirs"], np.uint8),
        "nums": np.array(state["nums"], np.uint32),
        "flags": np.array(state["flags"], np.uint8),
        "next": np.array(state["next"], np.int16),
        "prev": np.array(state["prev"], np.int16),
        "dsf_is_root": np.array(state["dsf"]["is_root"]),
        "dsf_parent": np.array(state["dsf"]["parent"], dtype=np.int32),
        "dsf_size": np.array(state["dsf"]["size"], dtype=np.int32),
        "dsf_opposite_parent": np.array(state["dsf"]["opposite_parent"]),
        "numsi": np.array(state["numsi"], np.int16),
        "dragging": get_binary_obs(state["dragging"]),
        "drag_is_from": get_binary_obs(state["drag_is_from"]),
        "drag_start": np.array([state["sx"], state["sy"]], np.int16),
    }


def get_observation_space_singles(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    maxwh = max(w, h)
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(wh, start=1),
        "o": Discrete(maxwh, start=1),
        "nums": Box(1, maxwh, (wh,), np.uint8),
        "flags": MultiDiscrete([9] * wh, np.uint8),
    }


def get_observation_singles(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "n": state["n"],
        "o": state["o"],
        "nums": np.array(state["nums"], np.uint8),
        "flags": np.array(state["flags"], np.uint8),
    }


def get_observation_space_sixteen(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": Box(-1, max(w, h), (2,), np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(wh, start=1),
        "tiles": Box(1, wh, (wh,), np.uint16),
        "movecount": Box(0, UINT_MAX, dtype=np.uint32),
        "movetarget": Box(0, UINT_MAX, dtype=np.uint32),
        "last_movement_sense": Discrete(3, start=-1),
    }


def get_observation_sixteen(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "n": state["n"],
        "tiles": np.array(state["tiles"], np.uint16),
        "movecount": np.array([state["movecount"]], np.uint32),
        "movetarget": np.array([state["movetarget"]], np.uint32),
        "last_movement_sense": state["last_movement_sense"],
    }


def get_observation_space_slant(game_state: GameState) -> dict[str, Space]:
    w = game_state.p.w
    h = game_state.p.h
    wh = w * h
    wphp = (w + 1) * (h + 1)
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "clues": Box(-1, 4, (wphp,), np.int8),
        "soln": Box(-1, 1, (wh,), np.int8),
        "errors": MultiDiscrete([2] * wphp, np.uint8),
    }


def get_observation_slant(state: dict) -> dict:
    return {
        "w": state["p"]["w"],
        "h": state["p"]["h"],
        "clues": np.array(state["clues"]["clues"], np.int8),
        "soln": np.array(state["soln"], np.int8),
        "errors": np.array(state["errors"], np.uint8),
    }


def get_observation_space_solo(game_state: GameState) -> dict[str, Space]:
    cr = game_state.cr
    c = game_state.blocks.contents.c
    r = game_state.blocks.contents.r
    area = cr * cr
    return {
        "cursor_pos": MultiDiscrete(nvec=[cr + 1, cr + 1], dtype=np.int32),
        "cr": Discrete(cr, start=1),
        "c": Discrete(c, start=1),
        "r": Discrete(r, start=1),
        "area": Discrete(area, start=1),
        "blocks": Dict(
            spaces={
                "whichblock": Box(0, area - 1, (area,), np.uint16),
                "blocks": Box(0, area - 1, (area,), np.uint16),
                "nr_squares": Box(0, area - 1, (area,), np.uint16),
                "nr_blocks": Discrete(area),
                "max_nr_squares": Discrete(area),
            }
        ),
        "kblocks": Dict(
            spaces={
                "whichblock": Box(0, area - 1, (area,), np.uint16),
                "blocks": Box(0, area - 1, (area,), np.uint16),
                "nr_squares": Box(0, area - 1, (area,), np.uint16),
                "nr_blocks": Discrete(area),
                "max_nr_squares": Discrete(area),
            }
        ),
        "xtype": Discrete(2),
        "killer": Discrete(2),
        "grid": Box(0, cr, (area,), np.uint8),
        "kgrid": Box(0, cr * area - 1, (area,), np.uint16),
        "pencil": MultiDiscrete([2] * area * cr, np.uint8),
        "immutable": MultiDiscrete([2] * area, np.uint8),
    }


def get_observation_solo(state: dict) -> dict:
    cr = state["cr"]
    c = state["blocks"]["c"]
    r = state["blocks"]["r"]
    area = cr * cr
    blocks_obj = state["blocks"]
    kblocks_blocks = np.zeros((area,), np.uint16)
    if state["killer"]:
        kblocks_obj = state["kblocks"]
        start = 0
        for i in range(len(kblocks_obj["nr_squares"])):
            kblocks_blocks[start : start + kblocks_obj["nr_squares"][i]] = kblocks_obj[
                "blocks"
            ][i]
            start += kblocks_obj["nr_squares"][i]

        kblock_nr_squares = np.zeros((area,), np.uint16)
        kblock_nr_squares[: kblocks_obj["nr_blocks"]] = kblocks_obj["nr_squares"]

        kblocks = {
            "whichblock": np.array(kblocks_obj["whichblock"], np.uint16),
            "blocks": kblocks_blocks,
            "nr_squares": kblock_nr_squares,
            "nr_blocks": kblocks_obj["nr_blocks"],
            "max_nr_squares": kblocks_obj["max_nr_squares"],
        }
        kgrid = np.array(state["kgrid"], np.uint16)
    else:
        kblocks = {
            "whichblock": np.zeros((area,), np.uint16),
            "blocks": kblocks_blocks,
            "nr_squares": np.zeros((area,), np.uint16),
            "nr_blocks": 0,
            "max_nr_squares": 0,
        }
        kgrid = np.zeros((area,), np.uint16)

    blocks_blocks = np.zeros((area,), np.uint16)
    start = 0
    for i in range(len(blocks_obj["nr_squares"])):
        blocks_blocks[start : start + blocks_obj["nr_squares"][i]] = blocks_obj[
            "blocks"
        ][i]
        start += blocks_obj["nr_squares"][i]
    blocks_nr_squares = np.zeros((area,), np.uint16)
    blocks_nr_squares[: blocks_obj["nr_blocks"]] = blocks_obj["nr_squares"]

    return {
        "cr": cr,
        "c": c,
        "r": r,
        "area": area,
        "blocks": {
            "whichblock": np.array(blocks_obj["whichblock"], np.uint16),
            "blocks": blocks_blocks,
            "nr_squares": blocks_nr_squares,
            "nr_blocks": blocks_obj["nr_blocks"],
            "max_nr_squares": blocks_obj["max_nr_squares"],
        },
        "kblocks": kblocks,
        "xtype": get_binary_obs(state["xtype"]),
        "killer": get_binary_obs(state["killer"]),
        "grid": np.array(state["grid"], np.uint8),
        "kgrid": kgrid,
        "pencil": np.array(state["pencil"], np.uint8),
        "immutable": np.array(state["immutable"], np.uint8),
    }


def get_observation_space_tents(game_state: GameState) -> dict[str, Space]:
    w = game_state.p.w
    h = game_state.p.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "grid": MultiDiscrete([4] * wh, np.uint8),
        "numbers": MultiDiscrete([max(w, h)] * (w + h), np.uint8),
    }


def get_observation_tents(state: dict) -> dict:
    w = state["p"]["w"]
    h = state["p"]["h"]
    return {
        "w": w,
        "h": h,
        "grid": np.array(state["grid"], np.uint8),
        "numbers": np.array(state["numbers"]["numbers"], np.uint8),
    }


def get_observation_space_towers(game_state: GameState) -> dict[str, Space]:
    w = game_state.par.w
    ww = w * w
    w4 = w * 4
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, w + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "clues": MultiDiscrete([10] * w4, np.uint8),
        "clues_done": MultiDiscrete([2] * w4, np.uint8),
        "grid": MultiDiscrete([w + 1] * ww, np.uint8),
        "immutable": MultiDiscrete([w + 1] * ww, np.uint8),
        "pencil": Box(0, 1023, (ww,), np.uint16),
    }


def get_observation_towers(state: dict) -> dict:
    return {
        "w": state["par"]["w"],
        "clues": np.array(state["clues"]["clues"], np.uint8),
        "clues_done": np.array(state["clues_done"], np.uint8),
        "grid": np.array(state["grid"], np.uint8),
        "immutable": np.array(state["clues"]["immutable"], np.uint8),
        "pencil": np.array(state["pencil"], np.uint16),
    }


def get_observation_space_tracks(game_state: GameState) -> dict[str, Space]:
    w = game_state.p.w
    h = game_state.p.h
    wph = w + h
    wh = w * h
    maxwh = max(w, h)

    return {
        "cursor_pos": MultiDiscrete(nvec=[2 * w, 2 * h], dtype=np.int32),
        "grid": Dict(
            spaces={
                "loop_ends": Box(low=0, high=wh, shape=(2,), dtype=np.int32),
                "set_directions": MultiBinary([wh, 4]),
                "set_connections": Box(low=0, high=4, shape=(wh,), dtype=np.uint8),
                "set_track": MultiBinary(wh),
                "no_directions": MultiBinary([wh, 4]),
                "no_connections": Box(low=0, high=4, shape=(wh,), dtype=np.uint8),
                "no_track": MultiBinary(wh),
                "is_clue": MultiBinary(wh),
            }
        ),
        "numbers": Box(low=0, high=maxwh, shape=(wph,), dtype=np.int32),
        "num_errors": MultiBinary(wph),
    }


def get_observation_tracks(state: dict) -> dict:
    w = state["p"]["w"]
    h = state["p"]["h"]
    wh = w * h

    loop_ends = np.asarray(state["grid"]["loop_ends"], dtype=np.int32)
    set_directions = np.zeros((wh, 4), dtype=np.uint8)
    set_connections = np.zeros((wh,), dtype=np.uint8)
    set_track = np.zeros((wh,), dtype=np.uint8)
    no_directions = np.zeros((wh, 4), dtype=np.uint8)
    no_connections = np.zeros((wh,), dtype=np.uint8)
    no_track = np.zeros((wh,), dtype=np.uint8)
    is_clue = np.zeros((wh,), dtype=np.uint8)

    for tile in range(wh):
        tile_info = state["grid"]["tiles"][tile]
        if tile_info["set_directions"]:
            set_directions[tile, 0] = get_binary_obs("R" in tile_info["set_directions"])
            set_directions[tile, 1] = get_binary_obs("U" in tile_info["set_directions"])
            set_directions[tile, 2] = get_binary_obs("L" in tile_info["set_directions"])
            set_directions[tile, 3] = get_binary_obs("D" in tile_info["set_directions"])
        set_connections[tile] = tile_info["set_connections"]
        set_track[tile] = get_binary_obs(tile_info["set_track"])
        if tile_info["no_directions"]:
            no_directions[tile, 0] = get_binary_obs("R" in tile_info["no_directions"])
            no_directions[tile, 1] = get_binary_obs("U" in tile_info["no_directions"])
            no_directions[tile, 2] = get_binary_obs("L" in tile_info["no_directions"])
            no_directions[tile, 3] = get_binary_obs("D" in tile_info["no_directions"])
        no_connections[tile] = tile_info["no_connections"]
        no_track[tile] = get_binary_obs(tile_info["no_track"])
        is_clue[tile] = get_binary_obs(tile_info["is_clue"])

    return {
        "grid": {
            "loop_ends": loop_ends,
            "set_directions": set_directions,
            "set_connections": set_connections,
            "set_track": set_track,
            "no_directions": no_directions,
            "no_connections": no_connections,
            "no_track": no_track,
            "is_clue": is_clue,
        },
        "numbers": np.asarray(state["numbers"]["numbers"], dtype=np.int32),
        "num_errors": np.asarray(state["num_errors"], dtype=np.uint8),
    }


def get_observation_space_twiddle(game_state: GameState) -> dict[str, Space]:
    w = game_state.w
    h = game_state.h
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "n": Discrete(game_state.n, start=1),
        "orientable": Discrete(2),
        "grid": MultiDiscrete([4 * wh + 1] * wh, np.uint16),
        "movecount": Box(0, UINT_MAX, dtype=np.uint32),
        "movetarget": Box(0, UINT_MAX, dtype=np.uint32),
        "lastx": Discrete(w + 1, start=-1),
        "lasty": Discrete(h + 1, start=-1),
        "lastr": Discrete(3, start=-1),
    }


def get_observation_twiddle(state: dict) -> dict:
    return {
        "w": state["w"],
        "h": state["h"],
        "n": state["n"],
        "orientable": get_binary_obs(state["n"]),
        "grid": np.array(state["grid"], np.uint16),
        "movecount": np.array([state["movecount"]], np.uint32),
        "movetarget": np.array([state["movetarget"]], np.uint32),
        "lastx": state["lastx"],
        "lasty": state["lasty"],
        "lastr": state["lastr"],
    }


def get_observation_space_undead(game_state: GameState) -> dict[str, Space]:
    w = game_state.common.contents.params.w
    h = game_state.common.contents.params.h
    wh = w * h
    wph = w + h
    extended_wh = game_state.common.contents.wh
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 3, h + 3], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "wh": Discrete(extended_wh, start=1),
        "num_ghosts": Discrete(wh + 1),
        "num_vampires": Discrete(wh + 1),
        "num_zombies": Discrete(wh + 1),
        "num_total": Discrete(wh + 1),
        "num_paths": Discrete(wph + 1),
        "paths": Dict(
            spaces={
                "length": MultiDiscrete([wh] * wph, np.uint16),
                "path": Box(-1, wh, (wph * wh,), np.int16),
                "xy": MultiDiscrete([extended_wh] * (wph * wh), np.uint16),
                "mapping": Box(-1, wh, (wph * wh,), np.int16),
                "grid_start": Box(-1, wph * 2, (wph,), np.int16),
                "grid_end": Box(-1, wph * 2, (wph,), np.int16),
                "num_monsters": MultiDiscrete([wh] * wph, np.uint16),
                "sightings_start": MultiDiscrete([wh] * wph, np.uint16),
                "sightings_end": MultiDiscrete([wh] * wph, np.uint16),
            }
        ),
        "grid": MultiDiscrete([7] * extended_wh, np.uint16),
        "xinfo": Box(-2, wh * 1, (extended_wh,), np.int16),
        "guess": MultiDiscrete([8] * wh, np.uint16),
        "pencils": MultiDiscrete([8] * wh, np.uint16),
        "cell_errors": MultiBinary(extended_wh),
        "hint_errors": MultiBinary(2 * wph),
        "hints_done": MultiBinary(2 * wph),
        "count_errors": MultiBinary(3),
    }


def get_observation_undead(state: dict) -> dict:
    common = state["common"]
    w = common["params"]["w"]
    h = common["params"]["h"]
    wh = w * h
    num_paths = common["num_paths"]
    paths_length = np.zeros((num_paths,), np.uint16)
    paths_path = np.zeros((num_paths * wh,), np.int16)
    paths_xy = np.zeros((num_paths * wh,), np.uint16)
    paths_mapping = np.zeros((num_paths * wh,), np.int16)
    paths_grid_start = np.zeros((num_paths,), np.int16)
    paths_grid_end = np.zeros((num_paths,), np.int16)
    paths_num_monsters = np.zeros((num_paths,), np.uint16)
    paths_sightings_start = np.zeros((num_paths,), np.uint16)
    paths_sightings_end = np.zeros((num_paths,), np.uint16)

    guess = np.zeros((wh,), np.uint16)
    guess[: common["num_total"]] = state["guess"]
    pencils = np.zeros((wh,), np.uint16)
    pencils[: common["num_total"]] = state["pencils"]

    for i in range(num_paths):
        iwh = i * wh
        path = common["paths"][i]

        paths_length[i] = length = path["length"]
        paths_path[iwh : iwh + length] = path["p"]
        paths_xy[iwh : iwh + length] = path["xy"]
        paths_mapping[iwh : iwh + sum(num > -1 for num in path["p"])] = path["mapping"]
        paths_grid_start[i] = path["grid_start"]
        paths_grid_end[i] = path["grid_end"]
        paths_num_monsters[i] = path["num_monsters"]
        paths_sightings_start[i] = path["sightings_start"]
        paths_sightings_end[i] = path["sightings_end"]

    return {
        "w": w,
        "h": h,
        "wh": common["wh"],
        "num_ghosts": common["num_ghosts"],
        "num_vampires": common["num_vampires"],
        "num_zombies": common["num_zombies"],
        "num_total": common["num_total"],
        "num_paths": num_paths,
        "paths": {
            "length": paths_length,
            "path": paths_path,
            "grid_start": paths_grid_start,
            "grid_end": paths_grid_end,
            "num_monsters": paths_num_monsters,
            "sightings_start": paths_sightings_start,
            "sightings_end": paths_sightings_end,
            "xy": paths_xy,
            "mapping": paths_mapping,
        },
        "grid": np.array(common["grid"], np.uint16),
        "xinfo": np.array(common["xinfo"], np.int16),
        "guess": guess,
        "pencils": pencils,
        "cell_errors": np.array(state["cell_errors"], np.uint8),
        "hint_errors": np.array(state["hint_errors"], np.uint8),
        "hints_done": np.array(state["hints_done"], np.uint8),
        "count_errors": np.array(state["count_errors"], np.uint8),
    }


def get_observation_space_unequal(game_state: GameState) -> dict[str, Space]:
    order = game_state.order
    o2 = order * order
    return {
        "cursor_pos": MultiDiscrete(nvec=[order + 1] * 2, dtype=np.int32),
        "order": Discrete(order, start=1),
        "mode": Discrete(2),
        "nums": MultiDiscrete([10] * o2, np.uint8),
        "hints": MultiDiscrete([2] * o2 * order, np.uint8),
        "flags": Box(0, 8092, (o2,), np.uint16),
    }


def get_observation_unequal(state: dict) -> dict:
    return {
        "order": state["order"],
        "mode": state["mode"],
        "nums": np.array(state["nums"], np.uint8),
        "hints": np.array(state["hints"], np.uint8),
        "flags": np.array(state["flags"], np.uint16),
    }


def get_observation_space_unruly(game_state: GameState) -> dict[str, Space]:
    w = game_state.w2
    h = game_state.h2
    wh = w * h
    return {
        "cursor_pos": MultiDiscrete(nvec=[w + 1, h + 1], dtype=np.int32),
        "w": Discrete(w, start=1),
        "h": Discrete(h, start=1),
        "unique": Discrete(2),
        "grid": MultiDiscrete([3] * wh, np.uint8),
        "immutable": MultiDiscrete([2] * wh, np.uint8),
    }


def get_observation_unruly(state: dict) -> dict:
    return {
        "w": state["w2"],
        "h": state["h2"],
        "unique": get_binary_obs(state["unique"]),
        "grid": np.array(state["grid"], np.uint8),
        "immutable": np.array(state["common"]["immutable"], np.uint8),
    }


def get_observation_space_untangle(game_state: GameState) -> dict[str, Space]:
    n = game_state.params.n
    max_edges = n * (n - 1)
    return {
        "cursor_pos": Box(-INT_MAX, INT_MAX, (2,), np.int32),
        "n": Discrete(n + 1),
        "pts": Dict(
            spaces={
                "x": Box(0, LONG_MAX, (n,), np.int64),
                "y": Box(0, LONG_MAX, (n,), np.int64),
                "d": Box(0, LONG_MAX, (n,), np.int64),
            }
        ),
        "edges": Box(-1, n, (max_edges,), np.int32),
    }


def get_observation_untangle(state: dict, cursor_pos: tuple[int, int] | None = None):
    n = state["params"]["n"]
    state["pts"]
    max_edges = n * (n - 1)

    x = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    d = np.zeros((n,), dtype=np.int64)
    edges = np.full((max_edges,), fill_value=-1, dtype=np.int32)

    for i in range(n):
        x[i] = state["pts"][i]["x"]
        y[i] = state["pts"][i]["y"]
        d[i] = state["pts"][i]["d"]
    for i, edge in enumerate(state["edges"]):
        edge_row_start = edge["a"] * (n - 1)
        edge_row_away = (edge["a"] - 1) * edge["a"] // 2
        edge_col = edge["b"] - edge["a"] - 1
        edge_no = edge_row_start - edge_row_away + edge_col
        edges[edge_no * 2 : (edge_no + 1) * 2] = [edge["a"], edge["b"]]

    return {
        "cursor_pos": np.asarray(
            cursor_pos if cursor_pos else state["cursor_pos"], dtype=np.int64
        ),
        "n": n,
        "pts": {
            "x": x,
            "y": y,
            "d": d,
        },
        "edges": edges,
    }


get_observation_space_methods: dict[str, Callable] = {
    "blackbox": get_observation_space_blackbox,
    "bridges": get_observation_space_bridges,
    "cube": get_observation_space_cube,
    "dominosa": get_observation_space_dominosa,
    "fifteen": get_observation_space_fifteen,
    "filling": get_observation_space_filling,
    "flip": get_observation_space_flip,
    "flood": get_observation_space_flood,
    "galaxies": get_observation_space_galaxies,
    "guess": get_observation_space_guess,
    "inertia": get_observation_space_inertia,
    "keen": get_observation_space_keen,
    "lightup": get_observation_space_lightup,
    "loopy": get_observation_space_loopy,
    "magnets": get_observation_space_magnets,
    "map": get_observation_space_map,
    "mines": get_observation_space_mines,
    "mosaic": get_observation_space_mosaic,
    "net": get_observation_space_net,
    "netslide": get_observation_space_netslide,
    "palisade": get_observation_space_palisade,
    "pattern": get_observation_space_pattern,
    "pearl": get_observation_space_pearl,
    "pegs": get_observation_space_pegs,
    "range": get_observation_space_range,
    "rect": get_observation_space_rect,
    "samegame": get_observation_space_samegame,
    "signpost": get_observation_space_signpost,
    "singles": get_observation_space_singles,
    "sixteen": get_observation_space_sixteen,
    "slant": get_observation_space_slant,
    "solo": get_observation_space_solo,
    "tents": get_observation_space_tents,
    "towers": get_observation_space_towers,
    "tracks": get_observation_space_tracks,
    "twiddle": get_observation_space_twiddle,
    "undead": get_observation_space_undead,
    "unequal": get_observation_space_unequal,
    "unruly": get_observation_space_unruly,
    "untangle": get_observation_space_untangle,
}

get_observation_methods: dict[str, Callable] = {
    "blackbox": get_observation_blackbox,
    "bridges": get_observation_bridges,
    "cube": get_observation_cube,
    "dominosa": get_observation_dominosa,
    "fifteen": get_observation_fifteen,
    "filling": get_observation_filling,
    "flip": get_observation_flip,
    "flood": get_observation_flood,
    "galaxies": get_observation_galaxies,
    "guess": get_observation_guess,
    "inertia": get_observation_inertia,
    "keen": get_observation_keen,
    "lightup": get_observation_lightup,
    "loopy": get_observation_loopy,
    "magnets": get_observation_magnets,
    "map": get_observation_map,
    "mines": get_observation_mines,
    "mosaic": get_observation_mosaic,
    "net": get_observation_net,
    "netslide": get_observation_netslide,
    "palisade": get_observation_palisade,
    "pattern": get_observation_pattern,
    "pearl": get_observation_pearl,
    "pegs": get_observation_pegs,
    "range": get_observation_range,
    "rect": get_observation_rect,
    "samegame": get_observation_samegame,
    "signpost": get_observation_signpost,
    "singles": get_observation_singles,
    "sixteen": get_observation_sixteen,
    "slant": get_observation_slant,
    "solo": get_observation_solo,
    "tents": get_observation_tents,
    "towers": get_observation_towers,
    "tracks": get_observation_tracks,
    "twiddle": get_observation_twiddle,
    "undead": get_observation_undead,
    "unequal": get_observation_unequal,
    "unruly": get_observation_unruly,
    "untangle": get_observation_untangle,
}
