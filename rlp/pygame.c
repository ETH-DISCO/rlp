/*
 * pygame.c: PyGame front end for Simon Tatham's puzzle collection.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1 /* for strcasestr */
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "../puzzles/puzzles.h"
#include "../puzzles/gtk.h" // for icons

typedef struct PyCalls
{
    PyObject *pDrawText, *pDrawRect, *pDrawCircle;
    PyObject *pDrawLine, *pDrawThickLine, *pDrawPolygon;
    PyObject *pClip, *pUnclip;
    PyObject *pAddFont, *pStringMetrics;
    PyObject *pBlitterNew, *pBlitterFree, *pBlitterSave, *pBlitterLoad;
    PyObject *pSetWindowTitle, *pSetWindowIcon;
} PyCalls;

struct frontend
{
    int w, h;
    midend *me; /* for painting outside puzzle area */
    int ncolours;
    struct colour *colours;
    bool timer_active;
    struct timeval last_time;
    int nfonts, fontsize;
    struct font *fonts;
    PyObject *surf;
    PyCalls *py_calls;
    int pwidth, pheight; /* pixmap size (w, h are area size */
};

typedef struct colour
{
    int red, green, blue;
} colour;

struct font
{
    PyObject *font;
    int type;
    int size;
};
#define ADD_CALLABLE_PYTHON_FUNCTION(py_obj, py_dict, func) \
    py_obj = PyDict_GetItemString(py_dict, (char *)func);   \
    if (!PyCallable_Check(py_obj))                          \
    {                                                       \
        PyErr_Print();                                      \
    }

PyCalls *get_py_calls()
{
    PyCalls *py_calls;
    py_calls = (PyCalls *)smalloc(sizeof(PyCalls));

    PyObject *pName, *pModule, *pDict;

    pName = PyUnicode_FromString((char *)"rlp.puzzle");
    if (pName == NULL)
        PyErr_Print();
    pModule = PyImport_Import(pName);
    if (pModule == NULL)
        PyErr_Print();
    pDict = PyModule_GetDict(pModule);
    if (pDict == NULL)
        PyErr_Print();

    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pDrawText, pDict, "py_draw_text")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pDrawRect, pDict, "py_draw_rect")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pDrawCircle, pDict, "py_draw_circle")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pDrawLine, pDict, "py_draw_line")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pDrawThickLine, pDict, "py_draw_thick_line")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pDrawPolygon, pDict, "py_draw_polygon")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pClip, pDict, "py_clip")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pUnclip, pDict, "py_unclip")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pAddFont, pDict, "py_add_font")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pStringMetrics, pDict, "py_string_metrics")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pBlitterNew, pDict, "py_blitter_new")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pBlitterFree, pDict, "py_blitter_free")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pBlitterSave, pDict, "py_blitter_save")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pBlitterLoad, pDict, "py_blitter_load")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pSetWindowTitle, pDict, "py_set_window_title")
    ADD_CALLABLE_PYTHON_FUNCTION(py_calls->pSetWindowIcon, pDict, "py_set_window_icon")

    Py_DECREF(pDict);
    Py_DECREF(pModule);
    Py_DECREF(pName);

    return py_calls;
}

void init_python()
{
    PyObject *sys, *path;

    // Initialize the Python Interpreter
    Py_Initialize();

    // Add the current directory to the python path
    sys = PyImport_ImportModule("sys");
    path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_FromString("../"));

    Py_DECREF(path);
    Py_DECREF(sys);
}

void finalize_python(frontend *fe)
{
    // Decrease ref count of the pygame drawing function objects
    Py_XDECREF(fe->py_calls->pDrawText);
    Py_XDECREF(fe->py_calls->pDrawText);
    Py_XDECREF(fe->py_calls->pDrawRect);
    Py_XDECREF(fe->py_calls->pDrawCircle);
    Py_XDECREF(fe->py_calls->pDrawLine);
    Py_XDECREF(fe->py_calls->pDrawThickLine);
    Py_XDECREF(fe->py_calls->pDrawPolygon);
    Py_XDECREF(fe->py_calls->pClip);
    Py_XDECREF(fe->py_calls->pUnclip);
    Py_XDECREF(fe->py_calls->pAddFont);
    Py_XDECREF(fe->py_calls->pStringMetrics);
    Py_XDECREF(fe->py_calls->pBlitterNew);
    Py_XDECREF(fe->py_calls->pBlitterFree);
    Py_XDECREF(fe->py_calls->pBlitterSave);
    Py_XDECREF(fe->py_calls->pBlitterLoad);
    Py_XDECREF(fe->py_calls->pSetWindowTitle);
    Py_XDECREF(fe->py_calls->pSetWindowIcon);

    // Clear smalloc'd memory
    sfree(fe->py_calls);

    // Clear the fonts' ref count
    for (int i = 0; i < fe->nfonts; i++)
    {
        Py_XDECREF(fe->fonts[i].font);
    }

    // Decrease ref count of pygame surface
    Py_XDECREF(fe->surf);
}

struct blitter
{
    PyObject *surf;
    int w, h;
};

bool savefile_read(void *wctx, void *buf, int len)
{
    FILE *fp = (FILE *)wctx;
    int ret;

    ret = fread(buf, 1, len, fp);
    return (ret == len);
}

struct savefile_write_ctx
{
    FILE *fp;
    int error;
};

static void savefile_write(void *wctx, const void *buf, int len)
{
    struct savefile_write_ctx *ctx = (struct savefile_write_ctx *)wctx;
    if (fwrite(buf, 1, len, ctx->fp) < len)
        ctx->error = errno;
}

void c_serialise_state(frontend *fe, PyObject *pStateFile)
{
    const char *name;
    name = PyUnicode_AsUTF8(pStateFile);

    if (name)
    {
        FILE *fp;

        fp = fopen(name, "w");

        struct savefile_write_ctx ctx;
        ctx.fp = fp;
        ctx.error = 0;
        midend_serialise(fe->me, savefile_write, &ctx);
        fclose(fp);
        if (ctx.error)
        {
            char boxmsg[512];
            fprintf(stderr, "Error writing save file: %.400s",
                    strerror(ctx.error));
        }
    }
}

void c_deserialise_state(frontend *fe, PyObject *pStateFile)
{
    const char *name;
    const char *err;

    name = PyUnicode_AsUTF8(pStateFile);

    if (name)
    {
        FILE *fp = fopen(name, "r");

        if (!fp)
        {
            fprintf(stderr, "Unable to open saved game file");
            return;
        }

        err = midend_deserialise(fe->me, savefile_read, fp);

        fclose(fp);

        if (err)
        {
            fprintf(stderr, "Failed to deserialise the saved game file");
            return;
        }

        midend_redraw(fe->me);
    }
}

void c_force_redraw(frontend *fe)
{
    midend_redraw(fe->me);
}

void snaffle_colours(frontend *fe)
{
    int i, ncolours;
    float *colours;
    bool *success;

    colours = midend_colours(fe->me, &ncolours);

    fe->ncolours = ncolours;
    fe->colours = snewn(ncolours, struct colour);
    for (i = 0; i < ncolours; i++)
    {
        fe->colours[i].red = colours[i * 3 + 0] * 0xFF;
        fe->colours[i].green = colours[i * 3 + 1] * 0xFF;
        fe->colours[i].blue = colours[i * 3 + 2] * 0xFF;
    }
}

void c_draw_text(PyCalls *py_calls, PyObject *surf, int x, int y, PyObject *font, colour clr,
                 const char *text)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("OiiO(iii)z", surf, x, y, font, clr.red, clr.green, clr.blue, (char *)text);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pDrawText, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void c_get_string_metrics(PyCalls *py_calls, PyObject *font, const char *text, int *width, int *height)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("Oz", font, text);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pStringMetrics, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    *width = PyLong_AsLong(PyList_GetItem(pResult, 0));
    *height = PyLong_AsLong(PyList_GetItem(pResult, 1));

    Py_DECREF(pResult);
}

PyObject *c_add_font(frontend *fe, int index, int fonttype, int fontsize)
{
    PyObject *pArgs;

    pArgs = Py_BuildValue("ii", fonttype, fontsize);
    PyErr_Print();

    fe->fonts[index].font = PyObject_CallObject(fe->py_calls->pAddFont, pArgs);
    Py_DECREF(pArgs);
    if (fe->fonts[index].font == NULL)
        PyErr_Print();
}

void pygame_draw_text(void *handle, int x, int y, int fonttype,
                      int fontsize, int align, int colour,
                      const char *text)
{
    frontend *fe = (frontend *)handle;
    int i;

    /* Find or create the font. */
    for (i = 0; i < fe->nfonts; i++)
        if (fe->fonts[i].type == fonttype && fe->fonts[i].size == fontsize)
            break;

    if (i == fe->nfonts)
    {
        if (fe->fontsize <= fe->nfonts)
        {
            fe->fontsize = fe->nfonts + 10;
            fe->fonts = sresize(fe->fonts, fe->fontsize, struct font);
        }

        fe->nfonts++;

        fe->fonts[i].font = c_add_font(fe, i, fonttype, fontsize);
        fe->fonts[i].type = fonttype;
        fe->fonts[i].size = fontsize;
    }
    int width, height;

    /*
     * Vertical height is always the font size in pixels,
     * but the horizontal width varies with the given string
     */
    c_get_string_metrics(fe->py_calls, fe->fonts[i].font, text, &width, &height);
    if (align & ALIGN_VCENTRE)
        y -= height / 2;
    else
        y -= height;

    if (align & ALIGN_HCENTRE)
        x -= width / 2;
    else if (align & ALIGN_HRIGHT)
        x -= width;

    /*
     * Actually draw the text.
     */
    c_draw_text(fe->py_calls, fe->surf, x, y, fe->fonts[i].font, fe->colours[colour], text);
}

void c_draw_rect(PyCalls *py_calls, PyObject *surf,
                 int x, int y, int w, int h, colour clr)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("Oiiii(iii)", surf, x, y, w, h, clr.red, clr.green, clr.blue);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pDrawRect, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void pygame_draw_rect(void *handle, int x, int y, int w, int h, int colour)
{
    frontend *fe = (frontend *)handle;
    c_draw_rect(fe->py_calls, fe->surf, x, y, w, h, fe->colours[colour]);
}

void c_draw_line(PyCalls *py_calls, PyObject *surf,
                 int x1, int y1, int x2, int y2, colour colour)
{

    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("Oiiii(iii)", surf, x1, y1, x2, y2,
                          colour.red, colour.green, colour.blue);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pDrawLine, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}
void pygame_draw_line(void *handle, int x1, int y1, int x2, int y2,
                      int colour)
{
    frontend *fe = (frontend *)handle;
    c_draw_line(fe->py_calls, fe->surf, x1, y1, x2, y2, fe->colours[colour]);
}

void c_draw_thick_line(PyCalls *py_calls, PyObject *surf, float thickness,
                       float x1, float y1, float x2, float y2, colour colour)
{

    PyObject *pArgs, *pResult;
    // TODO: Possibly better rounding of thickness, as pygame only supports ints
    pArgs = Py_BuildValue("Oiffff(iii)", surf, (int)(thickness + 0.5), x1, y1, x2, y2,
                          colour.red, colour.green, colour.blue);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pDrawThickLine, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void pygame_draw_thick_line(void *handle, float thickness, float x1, float y1, float x2, float y2, int colour)
{
    frontend *fe = (frontend *)handle;
    c_draw_thick_line(fe->py_calls, fe->surf, thickness, x1, y1, x2, y2, fe->colours[colour]);
}

void c_draw_polygon(PyCalls *py_calls, PyObject *surf,
                    const int *coords, int npoints, colour fillcolour, colour outlinecolour)
{
    PyObject *pNum1, *pNum2, *pPair, *pCoords, *pArgs, *pResult;
    pCoords = PyList_New(npoints);
    for (int i = 0; i < npoints; i++)
    {
        pNum1 = PyLong_FromLong(coords[2 * i]);
        pNum2 = PyLong_FromLong(coords[2 * i + 1]);
        pPair = Py_BuildValue("[OO]", pNum1, pNum2);
        PyList_SET_ITEM(pCoords, i, pPair);
    }

    pArgs = Py_BuildValue("OO(iii)(iii)", surf, pCoords,
                          fillcolour.red, fillcolour.green, fillcolour.blue,
                          outlinecolour.red, outlinecolour.green, outlinecolour.blue);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pDrawPolygon, pArgs);
    if (pResult == NULL)
        PyErr_Print();

    Py_DECREF(pResult);
    Py_DECREF(pArgs);
    Py_DECREF(pCoords);
    Py_DECREF(pPair);
    Py_DECREF(pNum2);
    Py_DECREF(pNum1);
}

void pygame_draw_poly(void *handle, const int *coords, int npoints,
                      int fillcolour, int outlinecolour)
{
    frontend *fe = (frontend *)handle;
    colour fill_clr;
    if (fillcolour >= 0)
    {
        fill_clr = fe->colours[fillcolour];
    }
    else
    {
        fill_clr.red = -1;
        fill_clr.green = -1;
        fill_clr.blue = -1;
    }
    c_draw_polygon(fe->py_calls, fe->surf, coords, npoints,
                   fill_clr, fe->colours[outlinecolour]);
}

void c_draw_circle(PyCalls *py_calls, PyObject *surf,
                   int x, int y, int radius, colour fillcolour, colour outlinecolour)
{

    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("Oiii(iii)(iii)", surf, x, y, radius,
                          fillcolour.red, fillcolour.green,
                          fillcolour.blue, outlinecolour.red,
                          outlinecolour.green, outlinecolour.blue);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pDrawCircle, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}
void pygame_draw_circle(void *handle, int cx, int cy, int radius,
                        int fillcolour, int outlinecolour)
{
    frontend *fe = (frontend *)handle;
    colour fill_clr;
    if (fillcolour >= 0)
    {
        fill_clr = fe->colours[fillcolour];
    }
    else
    {
        fill_clr.red = -1;
        fill_clr.green = -1;
        fill_clr.blue = -1;
    }
    c_draw_circle(fe->py_calls, fe->surf, cx, cy, radius,
                  fill_clr, fe->colours[outlinecolour]);
}

void c_clip(PyCalls *py_calls, PyObject *surf,
            int x, int y, int w, int h)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("Oiiii", surf, x, y, w, h);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pClip, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void pygame_clip(void *handle, int x, int y, int w, int h)
{
    frontend *fe = (frontend *)handle;
    // For some puzzles that do not take up the entire square window,
    // make sure the clipping area does not include the outside pixels.
    if (x + w >= fe->pwidth)
    {
        w = fe->pwidth - x - 1;
    }
    if (y + h >= fe->pheight)
    {
        h = fe->pheight - y - 1;
    }
    c_clip(fe->py_calls, fe->surf, x, y, w, h);
}

void c_unclip(PyCalls *py_calls, PyObject *surf, int w, int h)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("Oiiii", surf, 0, 0, w, h);
    PyErr_Print();

    pResult = PyObject_CallObject(py_calls->pClip, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void pygame_unclip(void *handle)
{
    frontend *fe = (frontend *)handle;
    c_unclip(fe->py_calls, fe->surf, fe->pwidth, fe->pheight);
}

// The things usually done by this function's equivalent
// in other frontends is independently handled by pygame.
// Nevertheless, it is required for the code to work
void pygame_draw_update(void *handle, int x, int y, int w, int h) {}
// The things usually done by this function's equivalent
// in other frontends is independently handled by pygame.
// Nevertheless, it is required for the code to work
void pygame_start_draw(void *handle) {}
// The things usually done by this function's equivalent
// in other frontends is independently handled by pygame.
// Nevertheless, it is required for the code to work
void pygame_end_draw(void *handle) {}

blitter *pygame_blitter_new(void *handle, int w, int h)
{
    frontend *fe = (frontend *)handle;
    blitter *bl = snew(blitter);
    bl->w = w;
    bl->h = h;

    PyObject *pArgs;

    pArgs = Py_BuildValue("ii", w, h);
    PyErr_Print();

    bl->surf = PyObject_CallObject(fe->py_calls->pBlitterNew, pArgs);
    Py_DECREF(pArgs);
    if (bl->surf == NULL)
        PyErr_Print();
    return bl;
}

void pygame_blitter_free(void *handle, blitter *bl)
{
    frontend *fe = (frontend *)handle;
    PyObject *pResult;

    PyErr_Print();

    pResult = PyObject_CallOneArg(fe->py_calls->pBlitterFree, bl->surf);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
    sfree(bl);
}

void pygame_blitter_save(void *handle, blitter *bl, int x, int y)
{
    frontend *fe = (frontend *)handle;
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("OOii", fe->surf, bl->surf, x, y);
    PyErr_Print();

    pResult = PyObject_CallObject(fe->py_calls->pBlitterSave, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void pygame_blitter_load(void *handle, blitter *bl, int x, int y)
{
    frontend *fe = (frontend *)handle;
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("OOii", fe->surf, bl->surf, x, y);
    PyErr_Print();

    pResult = PyObject_CallObject(fe->py_calls->pBlitterLoad, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

static const struct drawing_api pygame_drawing = {
    pygame_draw_text,
    pygame_draw_rect,
    pygame_draw_line,
    pygame_draw_poly,
    pygame_draw_circle,
    pygame_draw_update,
    pygame_clip,
    pygame_unclip,
    pygame_start_draw,
    pygame_end_draw,
    NULL, /* NULL. status_bar */
    pygame_blitter_new,
    pygame_blitter_free,
    pygame_blitter_save,
    pygame_blitter_load,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL, /* {begin,end}_{doc,page,puzzle} */
    NULL,
    NULL, /* line_width, line_dotted */
    NULL, /* text_fallback*/
#ifdef NO_THICK_LINE
    NULL,
#else
    pygame_draw_thick_line,
#endif
};

void c_set_window_title(frontend *fe, const char *title)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("z", title);
    PyErr_Print();

    pResult = PyObject_CallOneArg(fe->py_calls->pSetWindowTitle, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

void c_set_window_icon(frontend *fe, const char *icon_prefix)
{
    PyObject *pArgs, *pResult;

    pArgs = Py_BuildValue("z", icon_prefix);
    PyErr_Print();

    pResult = PyObject_CallOneArg(fe->py_calls->pSetWindowIcon, pArgs);
    Py_DECREF(pArgs);
    if (pResult == NULL)
        PyErr_Print();
    Py_DECREF(pResult);
}

enum
{
    ARG_EITHER,
    ARG_SAVE,
    ARG_ID
}; /* for argtype */

bool c_new_game(frontend *fe, char *arg, bool allow_undo)
{

    if (allow_undo)
    {
        /* Starts a new game with the state history cleared.
         * Useful for puzzles that support the "undo" functionality
         * but we do not want an undo upon a completed game.
         */
        midend_free(fe->me);
        fe->me = midend_new(fe, &thegame, &pygame_drawing, fe);
    }

    char errbuf[1024];
    const char *err;

    errbuf[0] = '\0';
    if (arg)
    {
        err = midend_game_id(fe->me, arg);
        if (err)
        {
            sprintf(errbuf, "Invalid game ID: %.800s", err);
        }
    }
    if (*errbuf)
    {
        fputs(errbuf, stderr);
        midend_free(fe->me);
        sfree(fe);
        return false;
    }
    else
    {
        midend_new_game(fe->me);
        midend_size(fe->me, &fe->pwidth, &fe->pheight, true, 2.0);
        midend_redraw(fe->me);
        return true;
    }
}

frontend *new_window(PyObject *surf, int w, int h, char *arg, int argtype)
{
    frontend *fe;
    fe = snew(frontend);
    memset(fe, 0, sizeof(frontend));

    fe->fonts = NULL;
    fe->nfonts = fe->fontsize = 0;

    fe->timer_active = false;

    fe->pwidth = fe->w = w;
    fe->pheight = fe->h = h;

    fe->me = midend_new(fe, &thegame, &pygame_drawing, fe);

    fe->surf = surf;
    Py_INCREF(surf);
    fe->py_calls = get_py_calls();

    c_set_window_title(fe, thegame.name);
    // The game icons should have been generated before
    // TODO: Possibly use the xpm char** data, similar to gtk
    if (n_xpm_icons)
    {
        c_set_window_icon(fe, thegame.htmlhelp_topic);
    }
    char errbuf[1024];
    if (arg)
    {
        const char *err;
        FILE *fp;

        errbuf[0] = '\0';

        switch (argtype)
        {
        case ARG_ID:
            err = midend_game_id(fe->me, arg);
            if (!err)
                midend_new_game(fe->me);
            else
                sprintf(errbuf, "Invalid game ID: %.800s", err);
            break;
        case ARG_SAVE:
            fp = fopen(arg, "r");
            if (!fp)
            {
                sprintf(errbuf, "Error opening file: %.800s", strerror(errno));
            }
            else
            {
                err = midend_deserialise(fe->me, savefile_read, fp);
                if (err)
                    sprintf(errbuf, "Invalid save file: %.800s", err);
                fclose(fp);
            }
            break;
        default /*case ARG_EITHER*/:
            /*
             * First try treating the argument as a game ID.
             */
            err = midend_game_id(fe->me, arg);
            if (!err)
            {
                /*
                 * It's a valid game ID.
                 */
                midend_new_game(fe->me);
            }
            else
            {
                FILE *fp = fopen(arg, "r");
                if (!fp)
                {
                    sprintf(errbuf, "Supplied argument is neither a game ID (%.400s)"
                                    " nor a save file (%.400s)",
                            err, strerror(errno));
                }
                else
                {
                    err = midend_deserialise(fe->me, savefile_read, fp);
                    if (err)
                        sprintf(errbuf, "%.800s", err);
                    fclose(fp);
                }
            }
            break;
        }
        if (*errbuf)
        {
            fputs(errbuf, stderr);
            midend_free(fe->me);
            sfree(fe);
            return NULL;
        }
    }
    else
    {
        midend_new_game(fe->me);
    }
    snaffle_colours(fe);

    midend_size(fe->me, &fe->pwidth, &fe->pheight, true, 2.0);
    pygame_clip(fe, 0, 0, fe->pwidth, fe->pheight);
    midend_redraw(fe->me);

    return fe;
}

void destroy_window(frontend *fe)
{
    deactivate_timer(fe);
    midend_free(fe->me);
    finalize_python(fe);
}

void process_key(frontend *fe, int x, int y, int keyval)
{
    if (keyval >= 0)
        midend_process_key(fe->me, x, y, keyval, NULL);
}

int game_status(frontend *fe)
{
    return midend_status(fe->me);
}

void get_random_seed(void **randseed, int *randseedsize)
{
    struct timeval *tvp = snew(struct timeval);
    gettimeofday(tvp, NULL);
    *randseed = (void *)tvp;
    *randseedsize = sizeof(struct timeval);
}

void timer_func(void *data, float millis)
{
    frontend *fe = (frontend *)data;

    if (fe->timer_active)
    {
        struct timeval now;
        float elapsed;
        gettimeofday(&now, NULL);
        if (millis > 0)
            elapsed = millis;
        else
            elapsed = ((now.tv_usec - fe->last_time.tv_usec) * 0.000001F +
                       (now.tv_sec - fe->last_time.tv_sec));

        midend_timer(fe->me, elapsed); /* may clear timer_active */
        fe->last_time = now;
    }
}

void deactivate_timer(frontend *fe)
{
    if (!fe)
        return; /* can happen due to --generate */
    fe->timer_active = false;
}

void activate_timer(frontend *fe)
{
    if (!fe)
        return; /* can happen due to --generate */
    if (!fe->timer_active)
    {
        gettimeofday(&fe->last_time, NULL);
    }
    fe->timer_active = true;
}

void fatal(const char *fmt, ...)
{
    va_list ap;

    fprintf(stderr, "fatal error: ");

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fprintf(stderr, "\n");
    exit(1);
}

void frontend_default_colour(frontend *fe, float *output)
{
    output[0] = output[1] = output[2] = 0.9F;
}
