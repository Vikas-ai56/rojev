"""Correct teardown for robosuite's offscreen render contexts (a robosuite bug, fixed here for every RoboCasa run).

robosuite's `MjRenderContext.__del__` (vendor/robosuite/robosuite/utils/binding_utils.py:200) frees its MuJoCo
rendering context (`mjr_freeContext`) in whatever OpenGL context is current. RoboCasa builds a new simulator on reset
and leaves the old one, with its render context, to the garbage collector; when a collection frees it (any
`gc.collect()`, or a later automatic one), its framebuffer and texture ids are deleted in the live simulator's
context, which uses the same ids, and every later camera image is noise or blank (measured 2026-10-22: neighbour-pixel
difference 70–80 vs 5 for a real image, from the first step after `gc.collect()`). `install()` replaces that teardown
with one that makes the context's own GL context current before freeing it and restores the previous one after.
Remove when robosuite frees in its own context.
"""
import glfw
from robosuite.utils.binding_utils import MjRenderContext


def _free_in_own_context(self) -> None:
    previous = glfw.get_current_context()
    own = getattr(self.gl_ctx, "_context", None)
    if own:
        glfw.make_context_current(own)
    self.con.free()
    self.gl_ctx.free()
    if previous and previous != own:
        glfw.make_context_current(previous)
    del self.con, self.gl_ctx, self.scn, self.cam, self.vopt, self.pert


def install() -> None:
    MjRenderContext.__del__ = _free_in_own_context
