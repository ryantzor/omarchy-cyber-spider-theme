"""Cyberpunk skateboard: glossy deck, neon edge + underglow, glowing wheels.

Board frame: origin at the wheel contact centre, +X nose, +Z up.
"""

import math

import bmesh
import bpy

from lib import emit_mat, hexrgb, link, mesh_obj, new_mat, principled

WHEEL_R = 0.42
DECK_Z = 1.05  # deck underside
DECK_T = 0.14
DECK_TOP = DECK_Z + DECK_T
DECK_L, DECK_W = 9.8, 2.65
TRUCK_X = 3.2
WHEEL_Y = 1.12


def deck_material():
    """Black grip on top, glossy purple/magenta graphic underneath."""
    m, nt, p = new_mat("deck")
    N, L = nt.nodes, nt.links
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Normal"], sep.inputs[0])
    up = N.new("ShaderNodeMath")
    up.operation = "GREATER_THAN"
    up.inputs[1].default_value = 0.5
    L.new(sep.outputs["Z"], up.inputs[0])

    sepo = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sepo.inputs[0])
    wave = N.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 0.6
    wave.inputs["Distortion"].default_value = 3
    L.new(tc.outputs["Object"], wave.inputs["Vector"])
    under = N.new("ShaderNodeValToRGB")
    under.color_ramp.elements[0].color = hexrgb("#3a0f6b")
    under.color_ramp.elements[1].color = hexrgb("#ff2bd6")
    L.new(wave.outputs["Fac"], under.inputs["Fac"])

    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    L.new(up.outputs[0], mix.inputs[0])
    L.new(under.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = hexrgb("#0c0b14")
    L.new(mix.outputs[2], p.inputs["Base Color"])
    rough = N.new("ShaderNodeMix")
    rough.data_type = "FLOAT"
    L.new(up.outputs[0], rough.inputs[0])
    rough.inputs[2].default_value = 0.15
    rough.inputs[3].default_value = 0.9
    L.new(rough.outputs[0], p.inputs["Roughness"])
    p.inputs["Coat Weight"].default_value = 0.6
    return m


def _deck_bm():
    bm = bmesh.new()
    nx, ny = 64, 10
    verts = {}
    for k, z in enumerate((0.0, DECK_T)):
        for i in range(nx + 1):
            for j in range(ny + 1):
                u = i / nx * 2 - 1
                v = j / ny * 2 - 1
                x = u * DECK_L / 2
                w = DECK_W / 2
                ax = abs(x) - (DECK_L / 2 - w)
                if ax > 0:
                    w = math.sqrt(max(0.0, (DECK_W / 2) ** 2 - ax * ax))
                y = v * max(w, 0.02)
                ex = abs(x) - TRUCK_X - 0.6
                kick = ex * ex * 0.32 if ex > 0 else 0.0
                verts[k, i, j] = bm.verts.new((x, y, DECK_Z + z + kick + 0.06 * v * v))
    for k in (0, 1):
        for i in range(nx):
            for j in range(ny):
                q = [verts[k, i, j], verts[k, i + 1, j], verts[k, i + 1, j + 1], verts[k, i, j + 1]]
                bm.faces.new(q if k else q[::-1])
    for i in range(nx):
        for j in (0, ny):
            q = [verts[0, i, j], verts[0, i + 1, j], verts[1, i + 1, j], verts[1, i, j]]
            bm.faces.new(q if j == 0 else q[::-1])
    for j in range(ny):
        for i in (0, nx):
            q = [verts[0, i, j], verts[0, i, j + 1], verts[1, i, j + 1], verts[1, i, j]]
            bm.faces.new(q if i == nx else q[::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm, verts, nx, ny


def _outline(verts, nx, ny):
    ring = [verts[1, i, 0].co.copy() for i in range(nx + 1)]
    return ring + [verts[1, i, ny].co.copy() for i in range(nx, -1, -1)]


def _edge_strip(ring, mat, root):
    """Neon tube tracing the deck outline (follows kick + rounded ends)."""
    cu = bpy.data.curves.new("deck_neon", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.045
    cu.bevel_resolution = 2
    sp = cu.splines.new("POLY")
    sp.points.add(len(ring) - 1)
    for p, c in zip(sp.points, ring):
        p.co = (c.x * 1.003, c.y * 1.02, c.z - 0.05, 1)
    sp.use_cyclic_u = True
    cu.materials.append(mat)
    ob = link(bpy.data.objects.new("deck_neon", cu))
    ob.parent = root
    return ob


class Board:
    def __init__(self):
        self.root = link(bpy.data.objects.new("board", None))
        bm, verts, nx, ny = _deck_bm()
        ring = _outline(verts, nx, ny)
        deck = mesh_obj("deck", bm, [deck_material()])
        bev = deck.modifiers.new("bev", "BEVEL")
        bev.width = 0.03
        bev.segments = 2
        deck.parent = self.root
        cyan = emit_mat("board_cyan", hexrgb("#2ef2ff"), 9)
        yel = emit_mat("board_yellow", hexrgb("#ffd23f"), 9)
        _edge_strip(ring, cyan, self.root)

        # yellow + cyan light bars on the underside graphic (like the reference)
        for y, mat in ((0.45, yel), (-0.45, cyan)):
            cu = bpy.data.curves.new("underbar", "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = 0.06
            sp = cu.splines.new("POLY")
            sp.points.add(1)
            sp.points[0].co = (-TRUCK_X + 0.7, y, DECK_Z - 0.02, 1)
            sp.points[1].co = (TRUCK_X - 0.7, y, DECK_Z - 0.02, 1)
            cu.materials.append(mat)
            ob = link(bpy.data.objects.new("underbar", cu))
            ob.parent = self.root

        # underglow light that paints the ground beneath the board
        ld = bpy.data.lights.new("underglow", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = DECK_L * 0.7, DECK_W * 0.6
        ld.energy = 1200
        ld.color = (0.2, 0.95, 1.0)
        glow = link(bpy.data.objects.new("underglow", ld))
        glow.location = (0, 0, DECK_Z - 0.1)
        glow.parent = self.root  # area lights point down -Z by default

        self._trucks()

    def _cyl(self, name, loc, r, depth, mat, parent, axis="Y", segs=32):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r, depth=depth)
        ob = mesh_obj(name, bm, [mat])
        if axis == "Y":
            ob.rotation_euler = (math.pi / 2, 0, 0)
        ob.location = loc
        ob.parent = parent
        bev = ob.modifiers.new("bev", "BEVEL")
        bev.width = min(r, depth) * 0.2
        bev.segments = 3
        return ob

    def _trucks(self):
        metal = principled("truck", hexrgb("#9aa0b8"), rough=0.2, Metallic=1.0)
        urethane = principled(
            "wheel", hexrgb("#d9f7ff"), rough=0.15, Transmission_Weight=0.6, Coat_Weight=0.5
        )
        hub = emit_mat("wheel_hub", hexrgb("#ff2bd6"), 10)
        self.wheels = []
        for sx in (-1, 1):
            x = sx * TRUCK_X
            bp = bmesh.new()
            bmesh.ops.create_cube(bp, size=1.0)
            ob = mesh_obj("baseplate", bp, [metal], smooth=False)
            ob.scale = (1.2, 0.85, 0.12)
            ob.location = (x, 0, DECK_Z - 0.06)
            ob.parent = self.root
            hg = bmesh.new()
            bmesh.ops.create_cube(hg, size=1.0)
            ob = mesh_obj("hanger", hg, [metal], smooth=False)
            ob.scale = (0.4, 2.0, 0.26)
            ob.location = (x, 0, DECK_Z - 0.45)
            ob.parent = self.root
            self._cyl("axle", (x, 0, WHEEL_R), 0.05, DECK_W * 0.98, metal, self.root)
            for sy in (-1, 1):
                # spin pivot: rotating this empty about local Y spins the wheel
                piv = link(bpy.data.objects.new("wheel_pivot", None))
                piv.location = (x, sy * WHEEL_Y, WHEEL_R)
                piv.parent = self.root
                self._cyl("wheel", (0, 0, 0), WHEEL_R, 0.38, urethane, piv)
                self._cyl("hub", (0, sy * 0.2, 0), WHEEL_R * 0.55, 0.03, hub, piv)
                # one hub spoke so the spin reads on camera
                sp = bmesh.new()
                bmesh.ops.create_cube(sp, size=1.0)
                ob = mesh_obj("spoke", sp, [metal], smooth=False)
                ob.scale = (WHEEL_R * 1.1, 0.02, 0.06)
                ob.location = (0, sy * 0.22, 0)
                ob.parent = piv
                self.wheels.append(piv)

    def spin(self, angle):
        for piv in self.wheels:
            piv.rotation_euler = (0, angle, 0)
