#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../../puzzles/puzzles.h"

static PyMethodDef puzzle_constants_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef puzzle_constants = {
    PyModuleDef_HEAD_INIT,
    "constants",
    "Constants defined in puzzle.h for use with the Puzzle Python Class",
    -1,
    puzzle_constants_methods,
};

PyMODINIT_FUNC PyInit_constants()
{
    PyObject *module = PyModule_Create(&puzzle_constants);
    PyModule_AddIntMacro(module, FONT_FIXED);
    PyModule_AddIntMacro(module, FONT_VARIABLE);
    PyModule_AddIntMacro(module, LEFT_BUTTON);
    PyModule_AddIntMacro(module, MIDDLE_BUTTON);
    PyModule_AddIntMacro(module, RIGHT_BUTTON);
    PyModule_AddIntMacro(module, LEFT_DRAG);
    PyModule_AddIntMacro(module, MIDDLE_DRAG);
    PyModule_AddIntMacro(module, RIGHT_DRAG);
    PyModule_AddIntMacro(module, LEFT_RELEASE);
    PyModule_AddIntMacro(module, MIDDLE_RELEASE);
    PyModule_AddIntMacro(module, RIGHT_RELEASE);
    PyModule_AddIntMacro(module, CURSOR_UP);
    PyModule_AddIntMacro(module, CURSOR_DOWN);
    PyModule_AddIntMacro(module, CURSOR_LEFT);
    PyModule_AddIntMacro(module, CURSOR_RIGHT);
    PyModule_AddIntMacro(module, CURSOR_SELECT);
    PyModule_AddIntMacro(module, CURSOR_SELECT2);
    PyModule_AddIntMacro(module, UI_LOWER_BOUND);
    PyModule_AddIntMacro(module, UI_QUIT);
    PyModule_AddIntMacro(module, UI_NEWGAME);
    PyModule_AddIntMacro(module, UI_SOLVE);
    PyModule_AddIntMacro(module, UI_UNDO);
    PyModule_AddIntMacro(module, UI_REDO);
    PyModule_AddIntMacro(module, UI_UPPER_BOUND);
    PyModule_AddIntMacro(module, MOD_CTRL);
    PyModule_AddIntMacro(module, MOD_SHFT);
    PyModule_AddIntMacro(module, MOD_NUM_KEYPAD);
    PyModule_AddIntMacro(module, MOD_MASK);
    return module;
}