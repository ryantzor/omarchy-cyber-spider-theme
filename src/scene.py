"""Cyber Spider - Blender (EEVEE) intro for the Omarchy theme.

A fuzzy cyber jumping spider skates up a kicker, boardslides a neon rail in a
shower of sparks, kickflips off the end, powerslides to a stop on the right
third of the frame and blinks at you. The last frame is the still wallpaper.

Usage (from this directory):
  blender -b -P scene.py -- still OUT.png [--scale S] [--samples N]
  blender -b -P scene.py -- frame F OUT.png [--scale S] [--samples N]
  blender -b -P scene.py -- video OUT.mp4 [--scale S] [--samples N]
"""

import math
import os
import subprocess
import sys
import tempfile

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim  # noqa: E402
from anim import F_LAND, F_POP, F_RAIL, F_STOP, FPS, FRAMES, X_STOP  # noqa: E402
from board import Board  # noqa: E402
from lib import ease_in_out, emit_mat, hexrgb, lerp, link, mesh_obj, seg  # noqa: E402
from set import aim, build_camera, build_set  # noqa: E402
from spider import Spider  # noqa: E402

W, H = 1920, 1080

CAM_END = (X_STOP - 3.6, -17.5, 2.6)
CAM_END_AIM = (X_STOP - 3.3, 0.0, 3.9)


# --------------------------------------------------------------- sparks
class Sparks:
    """Deterministic emissive spark streaks spraying from the rail contact."""

    def __init__(self, rng, n=110):
        self.mat = emit_mat("spark", hexrgb("#ffb347"), 40)
        hot = emit_mat("spark_hot", hexrgb("#fff1c1"), 60)
        self.items = []
        for i in range(n):
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=1.0)
            ob = mesh_obj("spark", bm, [hot if i % 4 == 0 else self.mat])
            f0 = rng.uniform(F_RAIL, F_POP - 1)
            v = Vector((rng.uniform(-0.9, -0.2), rng.uniform(-0.7, 0.5), rng.uniform(0.05, 0.6)))
            self.items.append((ob, f0, v, rng.uniform(5, 12), rng.uniform(0.03, 0.06)))
        ld = bpy.data.lights.new("spark_light", "POINT")
        ld.color = (1.0, 0.65, 0.3)
        ld.shadow_soft_size = 0.3
        self.light = link(bpy.data.objects.new("spark_light", ld))

    def at(self, f):
        g = 0.035
        for ob, f0, v, life, r in self.items:
            u = f - f0
            if 0 <= u <= life:
                p0 = anim.contact_point(f0)
                vel = v - Vector((0, 0, g * u))
                ob.location = p0 + v * u - Vector((0, 0, g * u * u / 2))
                ob.rotation_euler = vel.to_track_quat("Z", "Y").to_euler()
                fade = 1 - u / life
                ob.scale = (r * fade, r * fade, r * fade + vel.length * 0.9)
            else:
                ob.location = (0, 0, -50)
                ob.scale = (1e-4, 1e-4, 1e-4)
        on = anim.grinding(f)
        self.light.location = anim.contact_point(f) + Vector((0, -0.4, 0.3))
        self.light.data.energy = (700 + 400 * math.sin(f * 2.7) ** 2) if on else 0.0

    def keyframe(self, f):
        for ob, *_ in self.items:
            for path in ("location", "rotation_euler", "scale"):
                ob.keyframe_insert(path, frame=f)
        self.light.keyframe_insert("location", frame=f)
        self.light.data.keyframe_insert("energy", frame=f)


# --------------------------------------------------------------- camera
def _smooth_board(f):
    xs = zs = w = 0.0
    for i in range(14):
        x, z = anim.board_xz(max(1.0, f - i * 0.7))
        k = math.exp(-i * 0.22)
        xs, zs, w = xs + x * k, zs + z * k, w + k
    return xs / w, zs / w


def camera_at(cam, f):
    """Low dynamic tracking angle that settles into the hero shot."""
    bx, bz = _smooth_board(f)
    _, z = anim.board_xz(f)
    zz = 0.5 * bz + 0.5 * z
    pos = [bx - 2.5, -16.5 - 0.6 * zz + 1.5 * math.sin(f * 0.03), 1.4 + zz * 0.45]
    tgt = [bx + 1.5, 0.0, 3.4 + zz * 0.95]
    k = ease_in_out(seg(f, F_LAND - 4, F_STOP + 6))
    pos = [lerp(a, b, k) for a, b in zip(pos, CAM_END)]
    tgt = [lerp(a, b, k) for a, b in zip(tgt, CAM_END_AIM)]
    pos[2] += 0.08 * math.sin(f * 0.11)  # light handheld drift
    aim(cam, pos, tgt)
    x, z = anim.board_xz(f)
    cam.data.dof.focus_distance = (Vector((x, 0, z + 3)) - Vector(pos)).length


# --------------------------------------------------------------- render
def setup_render(scale, samples):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = int(W * scale)
    sc.render.resolution_y = int(H * scale)
    sc.render.resolution_percentage = 100
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAMES
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Punchy"
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    sc.render.hair_type = "STRAND"
    e = sc.eevee
    e.taa_render_samples = samples
    e.use_raytracing = True
    e.use_shadows = True
    e.use_fast_gi = True
    e.volumetric_samples = 48
    sc.render.image_settings.file_format = "PNG"


def build():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    rng = np.random.default_rng(5)
    build_set(rng)
    return Board(), Spider(), build_camera(), Sparks(rng)


def bake(objs):
    """Key every frame so EEVEE gets motion blur and frames render standalone."""
    board, spider, cam, sparks = objs
    moving = [board.root, spider.root, cam] + board.wheels + spider.eye_pivots
    moving += [s for leg in spider.legs for s in leg["segs"]]
    for f in range(1, FRAMES + 1):
        B, S, feet, wheel, blink = anim.state(float(f), CAM_END)
        board.root.matrix_world = B
        board.spin(wheel)
        spider.pose(S, feet)
        spider.blink(blink)
        camera_at(cam, float(f))
        sparks.at(float(f))
        for ob in moving:
            for path in ("location", "rotation_euler", "scale"):
                ob.keyframe_insert(path, frame=f)
        cam.data.dof.keyframe_insert("focus_distance", frame=f)
        sparks.keyframe(f)


def render_frame(path, f):
    sc = bpy.context.scene
    sc.frame_set(f)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def encode(frames_dir, out):
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-start_number", "1",
        "-i", os.path.join(frames_dir, "f_%04d.png"),
        "-vf", f"scale={W}:{H}:flags=lanczos",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", out,
    ])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"--scale": 1.0, "--samples": 64}
    for key in list(opts):
        if key in argv:
            i = argv.index(key)
            opts[key] = type(opts[key])(argv[i + 1])
            del argv[i:i + 2]
    if not argv or argv[0] not in ("still", "frame", "video"):
        print(__doc__)
        sys.exit(1)
    setup_render(opts["--scale"], opts["--samples"])
    objs = build()
    bake(objs)
    sc = bpy.context.scene
    if argv[0] == "still":
        sc.render.use_motion_blur = False
        render_frame(argv[1], FRAMES)
    elif argv[0] == "frame":
        render_frame(argv[2], int(argv[1]))
    else:
        tmp = tempfile.mkdtemp(prefix="cyber-spider-")
        sc.render.filepath = os.path.join(tmp, "f_")
        bpy.ops.render.render(animation=True)
        encode(tmp, argv[1])
        print("frames kept in", tmp)


if __name__ == "__main__":
    main()
