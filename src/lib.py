"""Shared helpers: math, mesh builders, materials, hair strands, neon."""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


# --------------------------------------------------------------- math
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def seg(t, t0, t1):
    return clamp((t - t0) / (t1 - t0))


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_in_out(p):
    return p * p * (3 - 2 * p)


def damped(tau, freq, decay):
    if tau <= 0:
        return 0.0
    return math.exp(-decay * tau) * math.sin(freq * tau)


def hexrgb(h):
    """sRGB hex -> linear RGBA tuple."""
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, 1.0)


def quat(yaw=0.0, pitch=0.0, roll=0.0):
    """Rz(yaw) @ Ry(-pitch) @ Rx(roll); pitch>0 lifts the +X nose."""
    qz = Quaternion((0, 0, 1), yaw)
    qy = Quaternion((0, 1, 0), -pitch)
    qx = Quaternion((1, 0, 0), roll)
    return qz @ qy @ qx


def mat4(pos, q):
    return Matrix.Translation(Vector(pos)) @ q.to_matrix().to_4x4()


# --------------------------------------------------------------- objects
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_obj(name, bm, mats=(), smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    return link(bpy.data.objects.new(name, me))


def superellipsoid_bm(center, radii, e=1.0, u=48, v=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1.0)
    for vert in bm.verts:
        c = vert.co
        b = [math.copysign(abs(x) ** e, x) for x in c]
        vert.co = Vector(
            (
                center[0] + b[0] * radii[0],
                center[1] + b[1] * radii[1],
                center[2] + b[2] * radii[2],
            )
        )
    return bm


def cone_bm(length, r0, r1, segs=20):
    """Tapered tube from z=0 (radius r0) to z=length (radius r1)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, segments=segs, radius1=r0, radius2=r1, depth=length
    )
    for vert in bm.verts:
        vert.co.z += length / 2
    return bm


def superellipsoid_samples(rng, n, center, radii, e=1.0):
    """Random surface points + normals on a superellipsoid."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    b = np.sign(u) * np.abs(u) ** e
    r = np.asarray(radii)
    pts = np.asarray(center) + b * r
    q = 2.0 / e
    g = np.sign(b) * np.abs(b) ** (q - 1) / r
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    return pts, g


def cone_samples(rng, n, length, r0, r1):
    th = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(0.02, 0.98, n)
    r = r0 + (r1 - r0) * t
    pts = np.stack([np.cos(th) * r, np.sin(th) * r, t * length], 1)
    nz = (r0 - r1) / length
    nrm = np.stack([np.cos(th), np.sin(th), np.full(n, nz)], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm, t


# --------------------------------------------------------------- hair
def strands(
    name,
    roots,
    normals,
    comb,
    lengths,
    colors,
    mat,
    rng,
    npts=5,
    bend=0.65,
    jitter=0.25,
    r_root=0.0035,
    r_tip=0.0006,
):
    """Build a Curves object of combed strands. All arrays (N,3)/(N,)."""
    n = len(roots)
    comb = np.broadcast_to(np.asarray(comb, float), (n, 3))
    pts = np.zeros((n, npts, 3))
    pts[:, 0] = roots
    noise = rng.normal(size=(n, 3)) * jitter
    step = (lengths / (npts - 1))[:, None]
    for k in range(1, npts):
        t = k / (npts - 1)
        d = normals * (1 - bend * t) + comb * (bend * t) + noise * t
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        pts[:, k] = pts[:, k - 1] + d * step

    cv = bpy.data.hair_curves.new(name)
    cv.add_curves([npts] * n)
    cv.position_data.foreach_set("vector", pts.reshape(-1).astype(np.float32))
    rad = np.tile(np.linspace(r_root, r_tip, npts), n).astype(np.float32)
    ra = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    ra.data.foreach_set("value", rad)
    col = np.ones((n, 4), np.float32)
    col[:, :3] = colors
    ca = cv.attributes.new("col", "FLOAT_COLOR", "CURVE")
    ca.data.foreach_set("color", col.reshape(-1))
    cv.materials.append(mat)
    return link(bpy.data.objects.new(name, cv))


# --------------------------------------------------------------- materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    return m, nt, nt.nodes["Principled BSDF"]


def principled(name, color, rough=0.5, **inputs):
    m, nt, p = new_mat(name)
    p.inputs["Base Color"].default_value = color
    p.inputs["Roughness"].default_value = rough
    for k, v in inputs.items():
        p.inputs[k.replace("_", " ")].default_value = v
    return m


def hair_material(name, rough=0.42, gain=1.0):
    """Strand colour from the per-curve 'col' attribute; darker at the root."""
    m, nt, p = new_mat(name)
    N = nt.nodes
    attr = N.new("ShaderNodeAttribute")
    attr.attribute_name = "col"
    attr.attribute_type = "GEOMETRY"
    info = N.new("ShaderNodeHairInfo")
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.35, 0.35, 0.35, 1)
    ramp.color_ramp.elements[1].color = (gain, gain, gain, 1)
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    L = nt.links
    L.new(info.outputs["Intercept"], ramp.inputs["Fac"])
    L.new(attr.outputs["Color"], mix.inputs[6])
    L.new(ramp.outputs["Color"], mix.inputs[7])
    L.new(mix.outputs[2], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    p.inputs["Specular IOR Level"].default_value = 0.7
    p.inputs["Sheen Weight"].default_value = 0.4
    p.inputs["Subsurface Weight"].default_value = 0.15
    L.new(mix.outputs[2], p.inputs["Subsurface Radius"])
    return m


def band_ramp_material(name, stops, rough=0.55, coat=0.3):
    """Leg-segment colour bands along generated Z (0 at hip end, 1 at tip)."""
    m, nt, p = new_mat(name)
    N = nt.nodes
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "LINEAR"
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = stops[0]
    els[1].position, els[1].color = stops[1]
    for pos, col in stops[2:]:
        e = els.new(pos)
        e.color = col
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    p.inputs["Coat Weight"].default_value = coat
    return m


# --------------------------------------------------------------- neon
def emit_mat(name, color, strength=8.0):
    """Pure emission material (neon tubes, signs, sparks)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type == "BSDF_PRINCIPLED":
            nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def tube(name, pts, radius, mat, res=12):
    """Round tube along a polyline / smooth path (list of xyz)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, xyz in zip(sp.points, pts):
        p.co = (*xyz, 1.0)
    cu.materials.append(mat)
    _ = res
    return link(bpy.data.objects.new(name, cu))


def box_bm(x0, x1, y0, y1, z0, z1):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(x1 - x0, y1 - y0, z1 - z0), verts=bm.verts)
    bmesh.ops.translate(
        bm, vec=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), verts=bm.verts
    )
    return bm
