import ctypes as c
import copy
from typing import Union

import rlp.specific_api as specific

CT_PTR = c.POINTER
CT_FUNC = c.CFUNCTYPE
CT_UCHAR = c.c_ubyte
CT_SCHAR = c.c_byte
CT_BOOL_PTR = CT_PTR(c.c_bool)


class PyCalls(c.Structure):
    pass


class Colour(c.Structure):
    pass


PYCALLS_PTR = CT_PTR(PyCalls)
COLOUR_PTR = CT_PTR(Colour)
FRONTEND_PTR = CT_PTR(specific.Frontend)


specific.RandomState._fields_ = [
    ("seedbuf", CT_UCHAR * 40),
    ("databuf", CT_UCHAR * 20),
    ("pos", c.c_int),
]

specific.Blitter._fields_ = [
    ("surf", c.py_object),
    ("w", c.c_int),
    ("h", c.c_int),
]


class DrawingApi(c.Structure):
    _fields_ = [
        (
            "draw_text",
            CT_FUNC(
                c.c_void_p,
                c.c_void_p,
                c.c_int,
                c.c_int,
                c.c_int,
                c.c_int,
                c.c_int,
                c.c_int,
                c.c_char_p,
            ),
        ),
        (
            "draw_rect",
            CT_FUNC(
                c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int
            ),
        ),
        (
            "draw_line",
            CT_FUNC(
                c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int
            ),
        ),
        (
            "draw_polygon",
            CT_FUNC(
                c.c_void_p, c.c_void_p, specific.CT_INT_PTR, c.c_int, c.c_int, c.c_int
            ),
        ),
        (
            "draw_circle",
            CT_FUNC(
                c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int
            ),
        ),
        (
            "draw_update",
            CT_FUNC(c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int),
        ),
        ("clip", CT_FUNC(c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int)),
        ("unclip", CT_FUNC(c.c_void_p, c.c_void_p)),
        ("start_draw", CT_FUNC(c.c_void_p, c.c_void_p)),
        ("end_draw", CT_FUNC(c.c_void_p, c.c_void_p)),
        ("status_bar", CT_FUNC(c.c_void_p, c.c_void_p, c.c_char_p)),
        ("blitter_new", CT_FUNC(specific.BLITTER_PTR, c.c_void_p, c.c_int, c.c_int)),
        ("blitter_free", CT_FUNC(c.c_void_p, c.c_void_p, specific.BLITTER_PTR)),
        (
            "blitter_save",
            CT_FUNC(c.c_void_p, c.c_void_p, specific.BLITTER_PTR, c.c_int, c.c_int),
        ),
        (
            "blitter_load",
            CT_FUNC(c.c_void_p, c.c_void_p, specific.BLITTER_PTR, c.c_int, c.c_int),
        ),
        ("begin_doc", CT_FUNC(c.c_void_p, c.c_void_p, c.c_int)),
        ("begin_page", CT_FUNC(c.c_void_p, c.c_void_p, c.c_int)),
        (
            "begin_puzzle",
            CT_FUNC(
                c.c_void_p,
                c.c_void_p,
                c.c_float,
                c.c_float,
                c.c_float,
                c.c_float,
                c.c_int,
                c.c_int,
                c.c_float,
            ),
        ),
        ("end_puzzle", CT_FUNC(c.c_void_p, c.c_void_p)),
        ("end_page", CT_FUNC(c.c_void_p, c.c_void_p, c.c_int)),
        ("end_doc", CT_FUNC(c.c_void_p, c.c_void_p)),
        ("line_width", CT_FUNC(c.c_void_p, c.c_void_p, c.c_float)),
        ("line_dotted", CT_FUNC(c.c_void_p, c.c_void_p, c.c_bool)),
        ("text_fallback", CT_FUNC(c.c_void_p, c.c_void_p, CT_PTR(c.c_char_p))),
        (
            "draw_thick_line",
            CT_FUNC(
                c.c_void_p,
                c.c_void_p,
                c.c_float,
                c.c_float,
                c.c_float,
                c.c_float,
                c.c_float,
                c.c_int,
            ),
        ),
    ]


PyCalls._fields_ = [
    ("pDrawText", c.POINTER(c.py_object)),
    ("pDrawRect", c.POINTER(c.py_object)),
    ("pDrawCircle", c.POINTER(c.py_object)),
    ("pDrawLine", c.POINTER(c.py_object)),
    ("pDrawThickLine", c.POINTER(c.py_object)),
    ("pDrawPolygon", c.POINTER(c.py_object)),
    ("pClip", c.POINTER(c.py_object)),
    ("pUnclip", c.POINTER(c.py_object)),
    ("pAddFont", c.POINTER(c.py_object)),
    ("pStringMetrics", c.POINTER(c.py_object)),
    ("pBlitterNew", c.POINTER(c.py_object)),
    ("pBlitterFree", c.POINTER(c.py_object)),
    ("pBlitterSave", c.POINTER(c.py_object)),
    ("pBlitterLoad", c.POINTER(c.py_object)),
    ("pSetWindowTitle", c.POINTER(c.py_object)),
    ("pSetWindowIcon", c.POINTER(c.py_object)),
]


class Font(c.Structure):
    _fields_ = [
        ("font", c.py_object),
        ("type", c.c_int),
        ("size", c.c_int),
    ]


class Timeval(c.Structure):
    _fields_ = [
        ("tv_sec", c.c_long),
        ("tv_usec", c.c_long),
    ]


class _ConfigItemString(c.Structure):
    _fields_ = [
        ("sval", c.c_char_p),
    ]


class _ConfigItemChoices(c.Structure):
    _fields_ = [
        ("choicenames", c.c_char_p),
        ("selected", c.c_int),
    ]


class _ConfigItemBoolean(c.Structure):
    _fields_ = [
        ("bval", c.c_bool),
    ]


class _ConfigItemUnion(c.Union):
    _fields_ = [
        ("string", _ConfigItemString),
        ("choices", _ConfigItemChoices),
        ("boolean", _ConfigItemBoolean),
    ]


class ConfigItem(c.Structure):
    _fields_ = [
        ("name", c.c_char_p),
        ("type", c.c_int),
        ("u", _ConfigItemUnion),
    ]


class PresetMenu(c.Structure):
    pass


Colour._fields_ = [
    ("red", c.c_int),
    ("green", c.c_int),
    ("blue", c.c_int),
]


specific.Frontend._fields_ = [
    ("w", c.c_int),
    ("h", c.c_int),
    ("me", specific.MIDEND_PTR),
    ("ncolours", c.c_int),
    ("colours", CT_PTR(Colour)),
    ("timer_active", c.c_bool),
    ("last_time", Timeval),
    ("nfonts", c.c_int),
    ("fontsize", c.c_int),
    ("fonts", CT_PTR(Font)),
    ("surf", c.py_object),
    ("py_calls", CT_PTR(PyCalls)),
    ("pwidth", c.c_int),
    ("pheight", c.c_int),
]


class Drawing(c.Structure):
    pass


class KeyLabel(c.Structure):
    _fields_ = [
        ("label", c.c_char_p),
        ("button", c.c_int),
    ]


class Game(c.Structure):
    _fields_ = [
        ("name", c.c_char_p),
        ("winhelp_topic", c.c_char_p),
        ("htmlhelp_topic", c.c_char_p),
        ("default_params", CT_FUNC(specific.GAMEPARAMS_PTR)),
        (
            "fetch_preset",
            CT_FUNC(
                c.c_bool,
                c.c_int,
                CT_PTR(CT_PTR(c.c_char)),
                CT_PTR(specific.GAMEPARAMS_PTR),
            ),
        ),
        ("preset_menu", CT_FUNC(CT_PTR(PresetMenu))),
        ("decode_params", CT_FUNC(None, specific.GAMEPARAMS_PTR, CT_PTR(c.c_char))),
        ("encode_params", CT_FUNC(c.c_char_p, specific.GAMEPARAMS_PTR, c.c_bool)),
        ("free_params", CT_FUNC(None, specific.GAMEPARAMS_PTR)),
        ("dup_params", CT_FUNC(specific.GAMEPARAMS_PTR, specific.GAMEPARAMS_PTR)),
        ("can_configure", c.c_bool),
        ("configure", CT_FUNC(CT_PTR(ConfigItem), specific.GAMEPARAMS_PTR)),
        ("custom_params", CT_FUNC(specific.GAMEPARAMS_PTR, CT_PTR(ConfigItem))),
        ("validate_params", CT_FUNC(c.c_char_p, specific.GAMEPARAMS_PTR, c.c_bool)),
        ("new_desc", CT_FUNC(c.c_char_p, specific.GAMEPARAMS_PTR, c.c_char_p)),
        (
            "validate_desc",
            CT_FUNC(
                c.c_char_p,
                specific.GAMEPARAMS_PTR,
                specific.RANDOMSTATE_PTR,
                CT_PTR(c.c_char_p),
                c.c_bool,
            ),
        ),
        (
            "new_game",
            CT_FUNC(
                specific.GAMESTATE_PTR,
                specific.MIDEND_PTR,
                specific.GAMEPARAMS_PTR,
                c.c_char_p,
            ),
        ),
        ("dup_game", CT_FUNC(specific.GAMESTATE_PTR, specific.GAMESTATE_PTR)),
        ("free_game", CT_FUNC(None, specific.GAMESTATE_PTR)),
        ("can_solve", c.c_bool),
        (
            "solve",
            CT_FUNC(
                c.c_char_p,
                specific.GAMESTATE_PTR,
                specific.GAMESTATE_PTR,
                c.c_char_p,
                CT_PTR(c.c_char_p),
            ),
        ),
        ("can_format_as_text_ever", c.c_bool),
        ("can_format_as_text_now", CT_FUNC(c.c_bool, specific.GAMEPARAMS_PTR)),
        ("text_format", CT_FUNC(c.c_char_p, specific.GAMESTATE_PTR)),
        ("new_ui", CT_FUNC(specific.GAMEUI_PTR, specific.GAMESTATE_PTR)),
        ("free_ui", CT_FUNC(None, specific.GAMEUI_PTR)),
        ("encode_ui", CT_FUNC(c.c_char_p, specific.GAMEUI_PTR)),
        ("decode_ui", CT_FUNC(None, specific.GAMEUI_PTR, c.c_char_p)),
        (
            "request_keys",
            CT_FUNC(CT_PTR(KeyLabel), specific.GAMEPARAMS_PTR, specific.CT_INT_PTR),
        ),
        (
            "changed_state",
            CT_FUNC(
                None,
                specific.GAMEUI_PTR,
                specific.GAMESTATE_PTR,
                specific.GAMESTATE_PTR,
            ),
        ),
        (
            "current_key_label",
            CT_FUNC(c.c_char_p, specific.GAMEUI_PTR, specific.GAMESTATE_PTR, c.c_int),
        ),
        (
            "interpret_move",
            CT_FUNC(
                c.c_char_p,
                specific.GAMESTATE_PTR,
                specific.GAMEUI_PTR,
                specific.GAMEDRAWSTATE_PTR,
                c.c_int,
                c.c_int,
                c.c_int,
            ),
        ),
        (
            "execute_move",
            CT_FUNC(specific.GAMESTATE_PTR, specific.GAMESTATE_PTR, c.c_char_p),
        ),
        ("preferred_tilesize", c.c_int),
        (
            "compute_size",
            CT_FUNC(None, specific.GAMEPARAMS_PTR, c.c_int, c.c_int, c.c_int),
        ),
        (
            "set_size",
            CT_FUNC(
                None,
                CT_PTR(Drawing),
                specific.GAMEDRAWSTATE_PTR,
                specific.GAMEPARAMS_PTR,
                c.c_int,
            ),
        ),
        (
            "colours",
            CT_FUNC(CT_PTR(c.c_float), specific.FRONTEND_PTR, specific.CT_INT_PTR),
        ),
        (
            "new_drawstate",
            CT_FUNC(
                specific.GAMEDRAWSTATE_PTR, CT_PTR(Drawing), specific.GAMESTATE_PTR
            ),
        ),
        ("free_drawstate", CT_FUNC(None, CT_PTR(Drawing), specific.GAMEDRAWSTATE_PTR)),
        (
            "redraw",
            CT_FUNC(
                None,
                CT_PTR(Drawing),
                specific.GAMEDRAWSTATE_PTR,
                specific.GAMESTATE_PTR,
                specific.GAMESTATE_PTR,
                c.c_int,
                specific.GAMEUI_PTR,
                c.c_float,
                c.c_float,
            ),
        ),
        (
            "anim_length",
            CT_FUNC(
                c.c_float,
                specific.GAMESTATE_PTR,
                specific.GAMESTATE_PTR,
                c.c_int,
                specific.GAMEUI_PTR,
            ),
        ),
        (
            "flash_length",
            CT_FUNC(
                c.c_float,
                specific.GAMESTATE_PTR,
                specific.GAMESTATE_PTR,
                c.c_int,
                specific.GAMEUI_PTR,
            ),
        ),
        (
            "get_cursor_location",
            CT_FUNC(
                None,
                specific.GAMEUI_PTR,
                specific.GAMEDRAWSTATE_PTR,
                specific.GAMESTATE_PTR,
                specific.GAMEPARAMS_PTR,
                specific.CT_INT_PTR,
                specific.CT_INT_PTR,
                specific.CT_INT_PTR,
                specific.CT_INT_PTR,
            ),
        ),
        ("status", CT_FUNC(c.c_int, specific.GAMESTATE_PTR)),
        ("can_print", c.c_bool),
        ("can_print_in_colour", c.c_bool),
        (
            "print_size",
            CT_FUNC(
                None, specific.GAMEPARAMS_PTR, CT_PTR(c.c_float), CT_PTR(c.c_float)
            ),
        ),
        ("print", CT_FUNC(None, CT_PTR(Drawing), specific.GAMESTATE_PTR, c.c_int)),
        ("wants_statusbar", c.c_bool),
        ("is_timed", c.c_bool),
        (
            "timing_state",
            CT_FUNC(c.c_bool, specific.GAMESTATE_PTR, specific.GAMEUI_PTR),
        ),
        ("flags", c.c_int),
    ]


class PresetMenuEntry(c.Structure):
    _fields_ = [
        ("title", c.c_char_p),
        ("params", specific.GAMEPARAMS_PTR),
        ("entries", CT_PTR(PresetMenu)),
    ]


PresetMenu._fields_ = [
    ("n_entries", c.c_int),
    ("entries_size", c.c_int),
    ("entries", PresetMenuEntry),
]


class MidendStateEntry(c.Structure):
    _fields_ = [
        ("state", specific.GAMESTATE_PTR),
        ("movestr", c.c_char_p),
        ("movetype", c.c_int),
    ]


class MidendSerialiseBuf(c.Structure):
    _fields_ = [
        ("buf", c.c_char_p),
        ("len", c.c_int),
        ("size", c.c_int),
    ]


specific.Midend._fields_ = [
    ("frontend", specific.FRONTEND_PTR),
    ("random", specific.RANDOMSTATE_PTR),
    ("ourgame", CT_PTR(Game)),
    ("preset_menu", CT_PTR(PresetMenu)),
    ("encoded_presets", CT_PTR(c.c_char_p)),
    ("n_encoded_presets", c.c_int),
    ("desc", c.c_char_p),
    ("privdesc", c.c_char_p),
    ("seedstr", c.c_char_p),
    ("aux_info", c.c_char_p),
    ("genmode", c.c_int),
    ("nstates", c.c_int),
    ("statesize", c.c_int),
    ("statepos", c.c_int),
    ("states", CT_PTR(MidendStateEntry)),
    ("newgame_undo", MidendSerialiseBuf),
    ("newgame_redo", MidendSerialiseBuf),
    ("newgame_can_store_undo", c.c_bool),
    ("params", specific.GAMEPARAMS_PTR),
    ("curparams", specific.GAMEPARAMS_PTR),
    ("drawstate", specific.GAMEDRAWSTATE_PTR),
    ("first_draw", c.c_bool),
    ("ui", specific.GAMEUI_PTR),
    ("oldstate", specific.GAMESTATE_PTR),
    ("anim_time", c.c_float),
    ("anim_pos", c.c_float),
    ("flash_time", c.c_float),
    ("flash_pos", c.c_float),
    ("dir", c.c_int),
    ("timing", c.c_bool),
    ("elapsed", c.c_float),
    ("laststatus", c.c_char_p),
    ("drawing", CT_PTR(Drawing)),
    ("pressed_mouse_button", c.c_int),
    ("preferred_tilesize", c.c_int),
    ("preferred_tilesize_dpr", c.c_int),
    ("tilesize", c.c_int),
    ("winwidth", c.c_int),
    ("winheight", c.c_int),
    ("game_id_change_notify_function", CT_FUNC(c.c_void_p, c.c_void_p)),
    ("game_id_change_notify_ctx", c.c_void_p),
]


class PrintColour(c.Structure):
    _fields_ = [
        ("hatch", c.c_int),
        ("hatch_when", c.c_int),
        ("r", c.c_float),
        ("g", c.c_float),
        ("b", c.c_float),
        ("grey", c.c_float),
    ]


Drawing._fields_ = [
    ("drawing_api", CT_PTR(DrawingApi)),
    ("handle", c.c_void_p),
    ("print_colour", CT_PTR(PrintColour)),
    ("ncolours", c.c_int),
    ("coloursize", c.c_int),
    ("scale", c.c_float),
    ("midend", specific.MIDEND_PTR),
    ("laststatus", c.c_char_p),
]


def make_hash(object) -> Union[tuple, int]:
    """
    Recursively makes a hash from a dictionary, list, tuple or set, containing hashable values.
    """
    if isinstance(object, (set, tuple, list)):
        return tuple([make_hash(e) for e in object])
    elif not isinstance(object, dict):
        return hash(object)

    object_copy = copy.deepcopy(object)

    for key, value in object_copy.items():
        if key in ["movecount", "move_count"]:
            continue
        object_copy[key] = make_hash(value)

    return hash(tuple(frozenset(sorted(object_copy.items()))))
