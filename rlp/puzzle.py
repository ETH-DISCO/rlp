import argparse
from collections.abc import Callable, Sequence
from typing import Mapping, Any
import ctypes as c
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # nopep8

import numpy as np
import pygame

from rlp import api, constants


def wrap_function(lib: c.PyDLL, funcname: str, restype, argtypes) -> Callable:
    """Simplify wrapping generic ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


class Puzzle:
    """
    A class representing a logic puzzle from
    Simon Tatham's Portable Puzzle Collection.
    """

    def __init__(
        self,
        puzzle: str,
        width: int = 512,
        height: int = 512,
        arg: str | None = None,
        headless: bool = False,
    ):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # TODO: Make library loading directory agnostic
        self._build_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"lib/"
        )
        self._lib = c.PyDLL(os.path.join(self._build_path, f"lib{puzzle}.so"))
        api.specific.set_api_structures(puzzle)
        self.puzzle_name = puzzle

        # Register C library functions as callable instance objects
        self._init_python = wrap_function(self._lib, "init_python", None, None)
        self._destroy_window = wrap_function(
            self._lib, "destroy_window", api.FRONTEND_PTR, None
        )

        self._new_window = wrap_function(
            self._lib,
            "new_window",
            api.FRONTEND_PTR,
            [c.py_object, c.c_int, c.c_int, c.c_char_p, c.c_int],
        )

        get_py_calls = wrap_function(self._lib, "get_py_calls", api.PYCALLS_PTR, None)
        self.calls = get_py_calls()

        # Text
        draw_text_prototype = c.PYFUNCTYPE(
            None,
            api.PYCALLS_PTR,
            c.py_object,
            c.py_object,
            c.c_int,
            c.c_int,
            api.Colour,
            c.c_char_p,
        )
        self._draw_text = draw_text_prototype(("c_draw_text", self._lib))
        # Rectangle
        draw_rect_prototype = c.PYFUNCTYPE(
            None,
            api.PYCALLS_PTR,
            c.py_object,
            c.c_int,
            c.c_int,
            c.c_int,
            c.c_int,
            api.Colour,
        )
        self._draw_rect = draw_rect_prototype(("c_draw_rect", self._lib))
        # Circle
        draw_circle_prototype = c.PYFUNCTYPE(
            None,
            api.PYCALLS_PTR,
            c.py_object,
            c.c_int,
            c.c_int,
            c.c_int,
            api.Colour,
            api.Colour,
        )
        self._draw_circle = draw_circle_prototype(("c_draw_circle", self._lib))
        # Line
        draw_line_prototype = c.PYFUNCTYPE(
            None,
            api.PYCALLS_PTR,
            c.py_object,
            c.c_int,
            c.c_int,
            c.c_int,
            c.c_int,
            api.Colour,
        )
        self._draw_line = draw_line_prototype(("c_draw_line", self._lib))
        # Thick Line
        draw_thick_line_prototype = c.PYFUNCTYPE(
            None,
            api.PYCALLS_PTR,
            c.py_object,
            c.c_float,
            c.c_float,
            c.c_float,
            c.c_float,
            c.c_float,
            api.Colour,
        )
        self._draw_thick_line = draw_thick_line_prototype(
            ("c_draw_thick_line", self._lib)
        )
        # Polygon
        draw_polygon_prototype = c.PYFUNCTYPE(
            None,
            api.PYCALLS_PTR,
            c.py_object,
            c.POINTER(c.c_int),
            c.c_int,
            api.Colour,
            api.Colour,
        )
        self._draw_polygon = draw_polygon_prototype(("c_draw_polygon", self._lib))

        # Clipping functions
        clip_prototype = c.PYFUNCTYPE(
            None, api.PYCALLS_PTR, c.py_object, c.c_int, c.c_int, c.c_int, c.c_int
        )
        self._clip = clip_prototype(("c_clip", self._lib))

        unclip_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR, c.py_object)
        self._unclip = unclip_prototype(("c_unclip", self._lib))

        timer_func_prototype = c.PYFUNCTYPE(None, api.FRONTEND_PTR, c.c_float)
        self._timer_func = timer_func_prototype(("timer_func", self._lib))

        self._process_key = wrap_function(
            self._lib,
            "process_key",
            c.c_bool,
            [api.FRONTEND_PTR, c.c_int, c.c_int, c.c_int],
        )

        self._game_status = wrap_function(
            self._lib, "game_status", c.c_int, [api.FRONTEND_PTR]
        )

        new_game_prototype = c.PYFUNCTYPE(
            c.c_bool, api.FRONTEND_PTR, c.c_char_p, c.c_bool
        )
        self._new_game = new_game_prototype(("c_new_game", self._lib))

        serialise_state_prototype = c.PYFUNCTYPE(
            None, api.specific.FRONTEND_PTR, c.py_object
        )
        self._serialise_state = serialise_state_prototype(
            ("c_serialise_state", self._lib)
        )

        deserialise_state_prototype = c.PYFUNCTYPE(
            None, api.specific.FRONTEND_PTR, c.py_object
        )
        self._deserialise_state = deserialise_state_prototype(
            ("c_deserialise_state", self._lib)
        )

        force_redraw_prototype = c.PYFUNCTYPE(None, api.specific.FRONTEND_PTR)
        self._force_redraw = force_redraw_prototype(("c_force_redraw", self._lib))

        # add the tree traversal function for the puzzle Untangle
        self._get_puzzle_state_helper = None
        if self.puzzle_name == "untangle":
            index234_prototype = c.PYFUNCTYPE(
                api.specific.EDGE_PTR, api.specific.TREE234_PTR, c.c_int
            )
            self._get_puzzle_state_helper = index234_prototype(("index234", self._lib))

        # actual setup of window & instance variables
        pygame.init()
        self._init_python()

        if not headless:
            self.screen = pygame.display.set_mode(size=(width, height))
        self.surf = pygame.Surface((width, height))

        self.arg = None
        if arg:
            self.arg = arg.encode()
            self.fe = self._new_window(
                self.surf, self.surf.get_width(), self.surf.get_height(), self.arg, 2
            )
        else:
            self.fe = self._new_window(
                self.surf, self.surf.get_width(), self.surf.get_height(), None, 0
            )

        if not self.fe:
            self.destroy()
            pygame.quit()
            quit(1)

        if self.puzzle_name in api.specific.state_has_ui_info:
            self._get_puzzle_state_helper = self.fe.contents.me.contents.ui
        self._serialised_ui_save = (
            api.specific.GameUi(),
            api.specific.get_ui_save_extras(self),
        )
        self.ui_save = (api.specific.GameUi(), api.specific.get_ui_save_extras(self))
        self.action_mask_config = api.specific.get_action_mask_configuration(
            self.puzzle_name, self.fe.contents.me.contents
        )

        self._shift = 0
        self._ctrl = 0
        self._alt = False
        self._mouse_pressed = (False, False, False)

    def destroy(self):
        self._destroy_window(self.fe)

    def draw_text(
        self,
        x: int,
        y: int,
        fonttype: int,
        fontsize: int,
        align: int,
        colour: api.Colour,
        text: str,
    ):
        self._draw_text(
            self.calls, self.surf, x, y, fonttype, fontsize, align, colour, text
        )

    def draw_rect(self, x: int, y: int, w: int, h: int, colour: api.Colour):
        self._draw_rect(self.calls, self.surf, x, y, w, h, colour)

    def draw_circle(
        self,
        x: int,
        y: int,
        radius: int,
        fillcolour: api.Colour,
        outlinecolour: api.Colour,
    ):
        self._draw_circle(
            self.calls, self.surf, x, y, radius, fillcolour, outlinecolour
        )

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, colour: api.Colour):
        self._draw_line(self.calls, self.surf, x1, y1, x2, y2, colour)

    def draw_thick_line(
        self, thickness, x1: int, y1: int, x2: int, y2: int, colour: api.Colour
    ):
        self._draw_thick_line(self.calls, self.surf, thickness, x1, y1, x2, y2, colour)

    def draw_polygon(
        self, points: Sequence[int], fillcolour: api.Colour, outlinecolour: api.Colour
    ):
        # cast "points" to a c_int array
        coords = (c.c_int * len(points))(*points)
        self._draw_polygon(
            self.calls, self.surf, coords, len(points) // 2, fillcolour, outlinecolour
        )

    def clip(self, x: int, y: int, w: int, h: int):
        self._clip(self.calls, self.surf, x, y, w, h)

    def unclip(self):
        self._unclip(self.calls, self.surf)

    def proceed_animation(self):
        if hasattr(self, "screen"):
            self._timer_func(self.fe, 0)
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        else:
            self._timer_func(self.fe, 1e5)

    def process_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            return self.process_key_event(event)
        elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
            return self.process_button_event(event)
        elif event.type == pygame.MOUSEMOTION:
            return self.process_motion_event()
        else:
            return True

    def process_key_event(self, event: pygame.event.Event) -> bool:
        return self.process_key(event.type, event.key, event.mod)

    def get_key_val(self, key: int, mod: int) -> int:
        self._shift = constants.MOD_SHFT if mod & pygame.KMOD_SHIFT else 0
        self._ctrl = constants.MOD_CTRL if mod & pygame.KMOD_CTRL else 0

        self._alt = True if mod & pygame.KMOD_ALT else False

        if key == pygame.K_UP:
            keyval = self._shift | self._ctrl | constants.CURSOR_UP
        elif key == pygame.K_KP8 or key == pygame.K_KP_8:
            keyval = constants.MOD_NUM_KEYPAD | ord("8")
        elif key == pygame.K_DOWN:
            keyval = self._shift | self._ctrl | constants.CURSOR_DOWN
        elif key == pygame.K_KP2 or key == pygame.K_KP_2:
            keyval = constants.MOD_NUM_KEYPAD | ord("2")
        elif key == pygame.K_LEFT:
            keyval = self._shift | self._ctrl | constants.CURSOR_LEFT
        elif key == pygame.K_KP4 or key == pygame.K_KP_4:
            keyval = constants.MOD_NUM_KEYPAD | ord("4")
        elif key == pygame.K_RIGHT:
            keyval = self._shift | self._ctrl | constants.CURSOR_RIGHT
        elif key == pygame.K_KP6 or key == pygame.K_KP_6:
            keyval = constants.MOD_NUM_KEYPAD | ord("6")
        elif key == pygame.K_KP7 or key == pygame.K_KP_7:
            keyval = constants.MOD_NUM_KEYPAD | ord("7")
        elif key == pygame.K_KP1 or key == pygame.K_KP_1:
            keyval = constants.MOD_NUM_KEYPAD | ord("1")
        elif key == pygame.K_KP9 or key == pygame.K_KP_9:
            keyval = constants.MOD_NUM_KEYPAD | ord("9")
        elif key == pygame.K_KP3 or key == pygame.K_KP_3:
            keyval = constants.MOD_NUM_KEYPAD | ord("3")
        elif key == pygame.K_KP0 or key == pygame.K_KP_0:
            keyval = constants.MOD_NUM_KEYPAD | ord("0")
        elif key == pygame.K_KP5 or key == pygame.K_KP_5:
            keyval = constants.MOD_NUM_KEYPAD | ord("5")
        elif (
            key == pygame.K_BACKSPACE
            or key == pygame.K_DELETE
            or key == pygame.K_KP_PERIOD
        ):
            keyval = ord("\177")
        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            keyval = constants.CURSOR_SELECT
        elif key == pygame.K_SPACE:
            keyval = constants.CURSOR_SELECT2
        elif key == pygame.K_z:
            keyval = constants.UI_REDO
        elif len(pygame.key.name(key)) == 1:
            keyval = ord(pygame.key.name(key))
        else:
            keyval = -1

        return keyval

    def process_key(self, type: int, key: int, mod: int) -> bool:
        if type == pygame.KEYUP:
            return True
        self._process_key(self.fe, 0, 0, self.get_key_val(key, mod))
        return True

    def process_button_event(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN and event.type != pygame.MOUSEBUTTONUP:
            return True

        # detect which button was actually changed
        new_mouse_pressed = pygame.mouse.get_pressed(num_buttons=3)[0:3]
        changed_button = -1
        for i in range(len(new_mouse_pressed)):
            if new_mouse_pressed[i] ^ self._mouse_pressed[i]:
                changed_button = i
        self._mouse_pressed = new_mouse_pressed

        # assign correct value for key processing
        if changed_button == 1 or self._shift:
            button = constants.MIDDLE_BUTTON
        elif changed_button == 2 or self._alt:
            button = constants.RIGHT_BUTTON
        elif changed_button == 0:
            button = constants.LEFT_BUTTON
        else:
            return False

        if event.type == pygame.MOUSEBUTTONUP and button >= constants.LEFT_BUTTON:
            button += constants.LEFT_RELEASE - constants.LEFT_BUTTON
        x, y = pygame.mouse.get_pos()
        self._process_key(self.fe, x, y, button)
        return True

    def process_motion_event(self):
        if self._mouse_pressed[1] or self._shift:
            button = constants.MIDDLE_DRAG
        elif self._mouse_pressed[0]:
            button = constants.LEFT_DRAG
        elif self._mouse_pressed[2] or self._alt:
            button = constants.RIGHT_DRAG
        else:
            return False

        x, y = pygame.mouse.get_pos()
        self._process_key(self.fe, x, y, button)
        return True

    def game_status(self) -> int:
        return self._game_status(self.fe)

    def get_puzzle_state(self, include_cursor: bool = False) -> dict:
        """
        Returns a dict containing the puzzle's internal state.
        The logical representation of the state is based on the one used by Simon Tatham's
        original puzzle code.
        """
        state_dict = api.specific.get_puzzle_state_dict(
            self.puzzle_name,
            self.fe.contents.me.contents.states[
                self.fe.contents.me.contents.nstates - 1
            ].state.contents,
            self._get_puzzle_state_helper,
        )
        if include_cursor and self.puzzle_name not in api.specific.ui_reset_never:
            state_dict.update(
                {
                    "cursor_pos": api.specific.get_cursor_coords(
                        self.puzzle_name, self.fe.contents.me.contents
                    ),
                }
            )
        return state_dict

    def valid_action_mask(
        self, action_space, modifiers_value, single_action_index: int | None = None
    ) -> tuple[np.ndarray, list[str | None]]:
        """
        Returns either a mask for currently valid actions and
        the strings corresponding to each action
        OR for a single action, whether it is valid and its
        corresponding move string.
        """
        return api.specific.get_valid_action_mask(
            self,
            action_space,
            self.fe.contents.me.contents,
            modifiers_value,
            self.action_mask_config,
            single_action_index,
        )

    def set_cursor_active(self) -> None:
        if self.puzzle_name not in api.specific.ui_reset_never:
            api.specific.set_cursor_active(self.fe.contents.me.contents.ui.contents)

    def new_game(self, allow_undo=False) -> None:
        okay = self._new_game(self.fe, self.arg, allow_undo)
        if not okay:
            quit(1)

        msg = api.c.c_char_p()
        self.can_solve = self.fe.contents.me.contents.ourgame.contents.can_solve
        if self.can_solve:
            if self.puzzle_name == "mines":
                solution_string = None
                self.can_solve = False
            else:
                solution_string = self.fe.contents.me.contents.ourgame.contents.solve(
                    self.fe.contents.me.contents.states[0].state,
                    self.fe.contents.me.contents.states[0].state,
                    self.fe.contents.me.contents.aux_info,
                    msg,
                ).decode("utf-8")[1:]
        else:
            solution_string = None
        self.solution = self.moves_left_to_solve = solution_string

    def check_move_against_solution(self, move_string) -> bool:
        retvalue = False
        if (
            self.moves_left_to_solve is not None
            and move_string in self.moves_left_to_solve
        ):
            retvalue = True
            output_string = ""
            str_list = self.moves_left_to_solve.split(";" + move_string)
            for element in str_list:
                output_string += element
            self.moves_left_to_solve = output_string
        return retvalue

    def serialise_state(
        self, save_file_folder: str, env_ident: int | None = None
    ) -> Mapping[str, Any]:
        return_dict: dict[str, Any] = {}
        os.makedirs(save_file_folder, exist_ok=True)
        return_dict["savefile"] = os.path.join(
            save_file_folder,
            f"{self.puzzle_name}{f'_{env_ident}' if env_ident else ''}.sgtp",
        )
        self._serialise_state(self.fe, return_dict["savefile"])
        if self.puzzle_name not in api.specific.ui_reset_never:
            api.specific.backup_ui(
                self.puzzle_name,
                self.fe.contents.me.contents.ui.contents,
                self._serialised_ui_save,
            )
        puzzle_dict = self.__dict__.copy()
        return_dict["puzzle_dict"] = {
            k: v
            for k, v in puzzle_dict.items()
            if not (callable(v) or k in ["fe", "calls", "screen", "surf", "_lib"])
        }
        return return_dict

    def deserialise_state(self, state_dict):
        self._deserialise_state(self.fe, state_dict["savefile"])
        self.__dict__.update(state_dict["puzzle_dict"])
        if self.puzzle_name not in api.specific.ui_reset_never:
            api.specific.reset_ui(
                self.puzzle_name,
                self.fe.contents.me.contents.ui.contents,
                self._serialised_ui_save,
            )
        self._force_redraw(self.fe)


def py_add_font(fonttype: int, fontsize: int) -> pygame.font.Font:
    fonttype_str = "monospace" if fonttype == constants.FONT_FIXED else "sans"
    font = pygame.font.SysFont(fonttype_str, fontsize, bold=True)
    return font


def py_string_metrics(font: pygame.font.Font, text: str) -> list[int]:
    width, height = font.size(text)
    return [width, height]


def py_draw_text(
    surf: pygame.Surface,
    x: int,
    y: int,
    font: pygame.font.Font,
    colour: tuple[int, int, int],
    text: str,
):
    text_surf = font.render(text, True, colour)
    surf.blit(text_surf, (x, y))


def py_draw_rect(
    surf: pygame.Surface, x: int, y: int, w: int, h: int, colour: tuple[int, int, int]
):
    pygame.draw.rect(surface=surf, color=colour, rect=[x, y, w, h], width=0)


def py_draw_circle(
    surf: pygame.Surface,
    x: int,
    y: int,
    radius: int,
    fillcolour: tuple[int, int, int],
    outlinecolour: tuple[int, int, int],
):
    if fillcolour != (-1, -1, -1):
        pygame.draw.circle(
            surface=surf,
            color=fillcolour,
            center=(x + 0.5, y + 0.5),
            radius=radius,
            width=0,
        )
    pygame.draw.circle(
        surface=surf,
        color=outlinecolour,
        center=(x + 0.5, y + 0.5),
        radius=radius,
        width=1,
    )


def py_draw_line(
    surf: pygame.Surface,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    colour: tuple[int, int, int],
):
    pygame.draw.line(
        surface=surf, color=colour, start_pos=(x1, y1), end_pos=(x2, y2), width=1
    )


def py_draw_thick_line(
    surf: pygame.Surface,
    thickness: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    colour: tuple[int, int, int],
):
    if thickness < 1.0:
        thickness = 1.0

    p1 = pygame.math.Vector2(x1, y1)
    p2 = pygame.math.Vector2(x2, y2)
    line = (p2 - p1).normalize()
    orth_line = pygame.math.Vector2(-line.y, line.x) * (thickness // 2)
    points = [p1 - orth_line, p1 + orth_line, p2 + orth_line, p2 - orth_line]
    pygame.draw.polygon(surf, colour, points)


def py_draw_polygon(
    surf: pygame.Surface,
    points: Sequence[Sequence[float]],
    fillcolour: tuple[int, int, int],
    outlinecolour: tuple[int, int, int],
):
    if fillcolour != (-1, -1, -1):
        pygame.draw.polygon(surf, fillcolour, points, 0)
    pygame.draw.polygon(surf, outlinecolour, points, 1)


def py_clip(surf: pygame.Surface, x: int, y: int, w: int, h: int):
    surf.set_clip(pygame.Rect(x, y, w, h))


def py_unclip(surf: pygame.Surface):
    surf.set_clip(None)


def py_blitter_new(w: int, h: int) -> pygame.Surface:
    return pygame.Surface((w, h))


def py_blitter_free(surf: pygame.Surface):
    del surf


def py_blitter_save(surf: pygame.Surface, blitter_surf: pygame.Surface, x: int, y: int):
    blit_area = pygame.Rect(x, y, blitter_surf.get_width(), blitter_surf.get_height())
    blitter_surf.blit(surf, (0, 0), blit_area)


def py_blitter_load(surf: pygame.Surface, blitter_surf: pygame.Surface, x: int, y: int):
    surf.blit(blitter_surf, (x, y))


def py_set_window_title(title: str):
    pygame.display.set_caption(title)


def py_set_window_icon(icon_prefix: str):
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f"lib/icons/{icon_prefix}-96d24.png",
    )
    icon = pygame.image.load(path)
    pygame.display.set_icon(icon)


def make_puzzle_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one of Simon Tatham's Portable Puzzles."
    )
    parser.add_argument(
        "-p",
        "--puzzle",
        metavar="PUZZLE_NAME",
        type=str,
        help="Choose which puzzle to play.",
        default="net",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Screen width & height in pixels.",
        default=(880, 880),
        nargs=2,
    )
    parser.add_argument(
        "-a",
        "--arg",
        type=str,
        metavar="PARAMETERS",
        help="Choose the parameters for a non-randomly generated puzzle.\n Format: 'params', 'params:description' or 'params#seed'\n Example:'9x9t4dh#12' for a 9x9 type 4 difficulty hard Loopy puzzle with seed 12.",
    )
    parser.add_argument(
        "-hl",
        "--headless",
        action="store_true",
        help="Set this to enable headless mode.",
    )
    parser.add_argument(
        "-undo",
        "--allowundo",
        action="store_true",
        help="Set this to enable an 'undo' action in puzzles that support it.",
    )
    return parser
