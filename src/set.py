"""Neon street set: wet asphalt, curbs, kicker, grind rail, city, lights, camera.

World frame: +X is the travel direction, the camera sits at -Y looking +Y.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from lib import box_bm, emit_mat, hexrgb, link, mesh_obj, new_mat, principled, tube

PAL = {
    "cyan": hexrgb("#2ef2ff"),
    "magenta": hexrgb("#ff2bd6"),
    "purple": hexrgb("#8a3cff"),
    "yellow": hexrgb("#ffd23f"),
    "orange": hexrgb("#ff8a3d"),
    "asphalt": hexrgb("#0b0a12"),
    "concrete": hexrgb("#1d1a2b"),
    "night": hexrgb("#07060d"),
}

# kicker ramp: rises from RAMP_X0 to the lip at RAMP_X1
RAMP_X0, RAMP_X1, RAMP_H, RAMP_W = -27.0, -18.0, 3.4, 10.0
# grind rail: straight top section between RAIL_X0..RAIL_X1 at height RAIL_Z
RAIL_X0, RAIL_X1, RAIL_Z, RAIL_R = -8.0, 14.0, 4.4, 0.22


def ramp_z(x):
    s = (x - RAMP_X0) / (RAMP_X1 - RAMP_X0)
    if s <= 0:
        return 0.0
    return RAMP_H * min(s, 1.0) ** 2


def ramp_slope(x):
    s = (x - RAMP_X0) / (RAMP_X1 - RAMP_X0)
    if s <= 0 or s > 1:
        return 0.0
    return math.atan(2 * RAMP_H * s / (RAMP_X1 - RAMP_X0))


# --------------------------------------------------------------- materials
def wet_asphalt():
    """Near-black asphalt with glossy puddles (noise-driven roughness)."""
    m, nt, p = new_mat("asphalt")
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    puddle = N.new("ShaderNodeTexNoise")
    puddle.inputs["Scale"].default_value = 0.025
    puddle.inputs["Detail"].default_value = 3
    L.new(tc.outputs["Object"], puddle.inputs["Vector"])
    pr = N.new("ShaderNodeValToRGB")
    pr.color_ramp.elements[0].position = 0.46
    pr.color_ramp.elements[0].color = (0.03, 0.03, 0.03, 1)
    pr.color_ramp.elements[1].position = 0.56
    pr.color_ramp.elements[1].color = (0.5, 0.5, 0.5, 1)
    L.new(puddle.outputs["Fac"], pr.inputs["Fac"])
    L.new(pr.outputs["Color"], p.inputs["Roughness"])

    grit = N.new("ShaderNodeTexNoise")
    grit.inputs["Scale"].default_value = 3.0
    grit.inputs["Detail"].default_value = 10
    L.new(tc.outputs["Object"], grit.inputs["Vector"])
    gc = N.new("ShaderNodeValToRGB")
    gc.color_ramp.elements[0].color = PAL["asphalt"]
    gc.color_ramp.elements[1].color = hexrgb("#262236")
    L.new(grit.outputs["Fac"], gc.inputs["Fac"])
    L.new(gc.outputs["Color"], p.inputs["Base Color"])

    # bump only on the dry parts so puddles stay mirror-flat
    inv = N.new("ShaderNodeMath")
    inv.operation = "MULTIPLY"
    inv.inputs[1].default_value = 0.35
    L.new(pr.outputs["Color"], inv.inputs[0])
    bmp = N.new("ShaderNodeBump")
    L.new(inv.outputs[0], bmp.inputs["Strength"])
    L.new(grit.outputs["Fac"], bmp.inputs["Height"])
    L.new(bmp.outputs["Normal"], p.inputs["Normal"])
    p.inputs["Specular IOR Level"].default_value = 0.7
    return m


def concrete(name, base, scale=0.3, rough=0.75):
    m, nt, p = new_mat(name)
    N, L = nt.nodes, nt.links
    tex = N.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = 8
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = tuple(c * 0.55 for c in base[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.6) for c in base[:3]) + (1,)
    L.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    L.new(ramp.outputs["Color"], p.inputs["Base Color"])
    bmp = N.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.3
    L.new(tex.outputs["Fac"], bmp.inputs["Height"])
    L.new(bmp.outputs["Normal"], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = rough
    return m


# --------------------------------------------------------------- ground
def build_ground():
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    bmesh.ops.scale(bm, vec=(260, 160, 1), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(10, 40, 0), verts=bm.verts)
    mesh_obj("ground", bm, [wet_asphalt()], smooth=False)


def _wave_path(x0, x1, y, amp, freq, phase, z=0.18, n=90):
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        pts.append((x, y + amp * math.sin(freq * x + phase), z))
    return pts


def build_curbs():
    """Glowing street curbs snaking across the ground (like the reference)."""
    cyan = emit_mat("neon_cyan", PAL["cyan"], 6)
    mag = emit_mat("neon_magenta", PAL["magenta"], 6)
    curb = concrete("curb", hexrgb("#0d0b16"))
    for name, y, amp, ph, mat in (
        ("curb_front_m", -11.0, 1.6, 0.0, mag),
        ("curb_front_c", -9.8, 1.5, 0.4, cyan),
        ("curb_back_m", 16.0, 3.0, 2.0, mag),
        ("curb_back_c", 17.4, 3.0, 2.3, cyan),
    ):
        path = _wave_path(-70, 80, y, amp, 0.07, ph)
        tube(name, path, 0.16, mat)
        # low concrete curb the neon strip sits on
        base = [(x, yy, 0.0) for x, yy, _ in path]
        tube(name + "_base", base, 0.14, curb).scale = (1, 1, 0.9)


def build_ramp():
    bm = bmesh.new()
    prof = []
    n = 24
    for i in range(n + 1):
        x = RAMP_X0 + (RAMP_X1 - RAMP_X0) * i / n
        prof.append((x, ramp_z(x)))
    prof += [(RAMP_X1 + 0.8, RAMP_H), (RAMP_X1 + 0.8, 0.0)]
    hw = RAMP_W / 2
    front = [bm.verts.new((x, -hw, z)) for x, z in prof]
    back = [bm.verts.new((x, hw, z)) for x, z in prof]
    bm.faces.new(front[::-1])
    bm.faces.new(back)
    m = len(prof)
    for i in range(m):
        j = (i + 1) % m
        bm.faces.new([front[i], front[j], back[j], back[i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = mesh_obj("ramp", bm, [concrete("ramp", hexrgb("#2a2340"))], smooth=False)
    bev = ob.modifiers.new("bev", "BEVEL")
    bev.width = 0.1
    bev.segments = 3
    bev.limit_method = "ANGLE"
    # neon side trims + glowing coping
    mag = bpy.data.materials["neon_magenta"]
    cyan = bpy.data.materials["neon_cyan"]
    for s in (-1, 1):
        trim = [(x, s * (hw + 0.05), z + 0.05) for x, z in prof[: n + 1]]
        tube("ramp_trim", trim, 0.08, mag)
    tube("coping", [(RAMP_X1 + 0.05, -hw, RAMP_H), (RAMP_X1 + 0.05, hw, RAMP_H)], 0.14, cyan)
    return ob


def build_rail():
    """Glowing cyan rail: straight top, rounded drop at both ends."""
    cyan = bpy.data.materials["neon_cyan"]
    pts = []
    r = 1.2
    pts.append((RAIL_X0 - r, 0, 0))
    for i in range(9):
        a = i / 8 * math.pi / 2
        pts.append((RAIL_X0 - r + r * (1 - math.cos(a)), 0, RAIL_Z - r + r * math.sin(a)))
    pts.append((RAIL_X1, 0, RAIL_Z))
    for i in range(1, 9):
        a = i / 8 * math.pi / 2
        pts.append((RAIL_X1 + r * math.sin(a), 0, RAIL_Z - r + r * math.cos(a)))
    pts.append((RAIL_X1 + r, 0, 0))
    pts.insert(1, (RAIL_X0 - r, 0, RAIL_Z - r))
    pts.insert(-1, (RAIL_X1 + r, 0, RAIL_Z - r))
    tube("rail", pts, RAIL_R, cyan)
    steel = principled("post", hexrgb("#3a3f55"), rough=0.3, Metallic=1.0)
    for x in np.linspace(RAIL_X0 + 3, RAIL_X1 - 3, 3):
        tube("rail_post", [(x, 0, 0), (x, 0, RAIL_Z - RAIL_R)], 0.11, steel)


# --------------------------------------------------------------- city
def _sign(text, loc, size, mat, rot_x=math.pi / 2):
    cu = bpy.data.curves.new("sign", "FONT")
    cu.body = text
    cu.size = size
    cu.align_x = "CENTER"
    cu.extrude = size * 0.03
    cu.materials.append(mat)
    ob = link(bpy.data.objects.new("sign", cu))
    ob.location = loc
    ob.rotation_euler = (rot_x, 0, 0)
    return ob


def build_city(rng):
    """Back wall of the skate spot, neon signs, and a hazy skyline."""
    wall_m = concrete("wall", hexrgb("#16132a"), scale=0.08)
    mesh_obj("wall", box_bm(-120, 140, 34, 37, 0, 14), [wall_m], smooth=False)
    mag = bpy.data.materials["neon_magenta"]
    cyan = bpy.data.materials["neon_cyan"]
    purp = emit_mat("neon_purple", PAL["purple"], 10)
    yel = emit_mat("neon_yellow", PAL["yellow"], 8)
    # strip lights along the wall top
    tube("wall_strip", [(-120, 33.8, 13.6), (140, 33.8, 13.6)], 0.12, purp)
    _sign("NEON SKATE", (-34, 33.7, 6.0), 4.2, cyan)
    _sign("CYBER SPIDER", (52, 33.7, 7.0), 4.0, mag)
    _sign("ネオン", (10, 33.7, 9.5), 2.6, yel)

    # skyline: dark towers with lit windows and neon roof strips
    tower = principled("tower", hexrgb("#0d0b18"), rough=0.6)
    win_mats = [emit_mat(f"win{i}", c, s) for i, (c, s) in enumerate(
        [(PAL["cyan"], 4), (PAL["magenta"], 4), (hexrgb("#ffe7b3"), 3), (PAL["purple"], 4)]
    )]
    x = -160.0
    while x < 190:
        w = rng.uniform(10, 26)
        h = rng.uniform(40, 140)
        y = rng.uniform(70, 130)
        mesh_obj("tower", box_bm(x, x + w, y, y + 14, 0, h), [tower], smooth=False)
        for _ in range(int(w * h / 60)):
            wx = rng.uniform(x + 1, x + w - 2.5)
            wz = rng.uniform(16, h - 3)
            m = win_mats[rng.choice(4, p=[0.35, 0.2, 0.35, 0.1])]
            mesh_obj("win", box_bm(wx, wx + 1.4, y - 0.2, y, wz, wz + 1.8), [m], smooth=False)
        if rng.random() < 0.5:
            m = (mag, cyan, purp)[rng.integers(3)]
            tube("roof", [(x, y - 0.3, h), (x + w, y - 0.3, h)], 0.35, m)
        x += w + rng.uniform(1, 6)


# --------------------------------------------------------------- world/lights
def build_world():
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Color"].default_value = hexrgb("#120d24")
    bg.inputs["Strength"].default_value = 0.15
    # thin haze so the neon blooms into the air
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Density"].default_value = 0.0005
    vol.inputs["Color"].default_value = hexrgb("#8f7ad6")
    out = nt.nodes["World Output"]
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    bpy.context.scene.world = world


def add_light(name, kind, loc, target, energy, color, size=10.0, spread=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color[:3]
    if kind == "AREA":
        ld.shape = "DISK"
        ld.size = size
    elif kind == "SPOT":
        ld.spot_size = spread or 0.6
        ld.spot_blend = 0.7
        ld.shadow_soft_size = size
    ob = link(bpy.data.objects.new(name, ld))
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


def build_lights():
    add_light("magenta_l", "AREA", (-40, -25, 18), (0, 0, 3), 12000, PAL["magenta"], 30)
    add_light("cyan_r", "AREA", (60, -20, 14), (20, 0, 3), 12000, PAL["cyan"], 30)
    add_light("rim", "AREA", (10, 45, 30), (10, 0, 4), 25000, hexrgb("#b9a6ff"), 40)
    add_light("street", "SPOT", (10, 5, 60), (10, 0, 0), 50000, hexrgb("#d9d2ff"), 8, 1.1)
    add_light("hero", "AREA", (40, -30, 12), (42, 0, 4), 2500, hexrgb("#ffffff"), 12)


# --------------------------------------------------------------- camera
def build_camera():
    cd = bpy.data.cameras.new("cam")
    cd.lens = 32
    cd.sensor_width = 36
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = 0.6
    cd.dof.aperture_blades = 0
    cd.clip_start = 0.1
    cd.clip_end = 2000
    cam = link(bpy.data.objects.new("cam", cd))
    bpy.context.scene.camera = cam
    return cam


def aim(cam, pos, target):
    pos, target = Vector(pos), Vector(target)
    q = (target - pos).to_track_quat("-Z", "Y")
    cam.matrix_world = Matrix.Translation(pos) @ q.to_matrix().to_4x4()
    cam.data.dof.focus_distance = (target - pos).length


# --------------------------------------------------------------- compositor
def build_glare():
    """Bloom via the compositor Glare node (Blender 4.2+ removed EEVEE bloom)."""
    sc = bpy.context.scene
    ng = bpy.data.node_groups.new("comp", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N, L = ng.nodes, ng.links
    rl = N.new("CompositorNodeRLayers")
    gl = N.new("CompositorNodeGlare")
    gl.inputs["Type"].default_value = "Bloom"
    gl.inputs["Quality"].default_value = "High"
    gl.inputs["Threshold"].default_value = 0.9
    gl.inputs["Strength"].default_value = 0.6
    gl.inputs["Size"].default_value = 0.7
    out = N.new("NodeGroupOutput")
    L.new(rl.outputs["Image"], gl.inputs["Image"])
    L.new(gl.outputs["Image"], out.inputs[0])
    sc.compositing_node_group = ng


def build_set(rng):
    build_world()
    build_ground()
    build_curbs()
    build_ramp()
    build_rail()
    build_city(rng)
    build_lights()
    build_glare()
