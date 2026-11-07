"""Cute cartoony cyber jumping spider: fuzzy body, giant eyes, neon visor, IK legs.

Spider local frame: +X forward, +Z up. One unit ~ the carapace radius.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from lib import (
    band_ramp_material,
    cone_bm,
    cone_samples,
    emit_mat,
    hair_material,
    hexrgb,
    link,
    mesh_obj,
    principled,
    strands,
    superellipsoid_bm,
    superellipsoid_samples,
)

CEPH_C, CEPH_R, CEPH_E = (0.0, 0.0, 0.0), (0.92, 0.86, 0.74), 0.85
ABD_C, ABD_R = (-1.25, 0.0, 0.18), (0.72, 0.6, 0.55)

# hip (x, y) on the carapace rim, femur/tibia/tarsus lengths, base radius
LEGS = [
    ((0.45, 0.62), (1.45, 1.30, 0.62), 0.13),
    ((0.15, 0.72), (1.30, 1.20, 0.58), 0.12),
    ((-0.18, 0.72), (1.32, 1.22, 0.60), 0.12),
    ((-0.46, 0.62), (1.45, 1.32, 0.68), 0.12),
]
HIP_Z = -0.3

# foot targets in spider frame (right side, y>0): riding stance and air tuck
STANCE = [(2.2, 1.45), (0.95, 2.0), (-0.85, 2.0), (-2.2, 1.5)]
TUCK = [(1.2, 1.0), (0.45, 1.15), (-0.4, 1.15), (-1.15, 1.0)]

# (x, y, z, r): huge front pair (AME), smaller side pair (ALE)
EYES = [(0.74, 0.34, 0.08, 0.36), (0.5, 0.74, 0.24, 0.16)]

C = {
    "fur": hexrgb("#3d2470"),
    "fur_hi": hexrgb("#8a5cd6"),
    "fur_dk": hexrgb("#1c0f35"),
    "cyan": hexrgb("#2ef2ff"),
    "magenta": hexrgb("#ff2bd6"),
}


def col3(c):
    return np.array(c[:3])


def _fur_cols(rng, n):
    t = rng.random((n, 1))
    cols = col3(C["fur_dk"]) * (1 - t) + col3(C["fur_hi"]) * t
    return cols * 0.6 + col3(C["fur"]) * 0.4


def torus_bm(R, r, major=40, minor=10):
    bm = bmesh.new()
    rows = []
    for i in range(major):
        a = 2 * math.pi * i / major
        row = []
        for j in range(minor):
            b = 2 * math.pi * j / minor
            row.append(bm.verts.new((
                (R + r * math.cos(b)) * math.cos(a),
                (R + r * math.cos(b)) * math.sin(a),
                r * math.sin(b),
            )))
        rows.append(row)
    for i in range(major):
        for j in range(minor):
            a, b = rows[i][j], rows[(i + 1) % major][j]
            c, d = rows[(i + 1) % major][(j + 1) % minor], rows[i][(j + 1) % minor]
            bm.faces.new([a, b, c, d])
    return bm


class Spider:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.root = link(bpy.data.objects.new("spider", None))
        self.hair_m = hair_material("spider_hair", rough=0.5, gain=1.2)
        self.skin = principled("spider_skin", C["fur_dk"], rough=0.5, Coat_Weight=0.3)
        self.cyan = emit_mat("spider_cyan", C["cyan"], 12)
        self.magenta = emit_mat("spider_magenta", C["magenta"], 10)
        self.eye_pivots = []
        self._body()
        self._eyes()
        self._visor()
        self.legs = [self._leg(i, s) for i in range(4) for s in (1, -1)]

    def _attach(self, obj, parent=None):
        obj.parent = parent or self.root
        return obj

    # ---------------------------------------------------------- body
    def _body(self):
        rng = self.rng
        self._attach(mesh_obj("ceph", superellipsoid_bm(CEPH_C, CEPH_R, CEPH_E), [self.skin]))
        self._attach(mesh_obj("abd", superellipsoid_bm(ABD_C, ABD_R, 1.0), [self.skin]))
        self._attach(
            mesh_obj("pedicel", superellipsoid_bm((-0.75, 0, 0.05), (0.3, 0.3, 0.28)), [self.skin])
        )
        eye_c = [np.array((x, s * y, z)) for x, y, z, _ in EYES for s in (1, -1)]
        eye_r = [r for *_, r in EYES for _ in (1, -1)]
        for name, c, r, e, cnt, comb_x in (
            ("ceph_fur", CEPH_C, CEPH_R, CEPH_E, 32000, -1.0),
            ("abd_fur", ABD_C, ABD_R, 1.0, 18000, -1.0),
        ):
            p, n = superellipsoid_samples(rng, cnt, c, r, e)
            keep = np.ones(len(p), bool)
            for ec, er in zip(eye_c, eye_r):
                keep &= np.linalg.norm(p - ec, axis=1) > er * 1.1
            p, n = p[keep], n[keep]
            comb = np.stack([np.full(len(p), comb_x), np.sign(p[:, 1]) * 0.4, np.full(len(p), -0.2)], 1)
            self._attach(
                strands(name, p, n, comb, rng.uniform(0.08, 0.16, len(p)), _fur_cols(rng, len(p)),
                        self.hair_m, rng, bend=0.8, jitter=0.3, r_root=0.011, r_tip=0.003)
            )
        # sparse long tufts for the cartoony fluffy silhouette
        p, n = superellipsoid_samples(rng, 1500, CEPH_C, CEPH_R, CEPH_E)
        up = n[:, 2] > 0.0
        self._attach(
            strands("tufts", p[up], n[up], (-0.5, 0, 0.6), rng.uniform(0.18, 0.3, up.sum()),
                    _fur_cols(rng, up.sum()) * 1.3, self.hair_m, rng,
                    npts=6, bend=0.35, jitter=0.4, r_root=0.008, r_tip=0.001)
        )

    # ---------------------------------------------------------- eyes
    def _eyes(self):
        glass = principled(
            "eye", hexrgb("#030308"), rough=0.03, Coat_Weight=1.0,
            Coat_Roughness=0.0, Specular_IOR_Level=1.0,
        )
        hl = emit_mat("eye_glint", (1, 1, 1, 1), 6)
        for x, y, z, r in EYES:
            for s in (1, -1):
                c = Vector((x, s * y, z))
                # pivot at the eye centre: scaling its local Z = blink
                piv = self._attach(link(bpy.data.objects.new("eye_pivot", None)))
                piv.location = c
                self.eye_pivots.append(piv)
                self._attach(mesh_obj("eye", superellipsoid_bm((0, 0, 0), (r, r, r), 1.0, 40, 24), [glass]), piv)
                # cyan iris ring on the outward-facing front of the eyeball
                out = Vector((1.0, s * (0.35 if r > 0.2 else 0.9), 0.15)).normalized()
                ring = mesh_obj("iris", torus_bm(r * 0.62, r * 0.07), [self.cyan])
                ring.matrix_basis = Matrix.Translation(out * r * 0.8) @ out.to_track_quat("Z", "Y").to_matrix().to_4x4()
                self._attach(ring, piv)
                # cartoon catchlight
                g = mesh_obj("glint", superellipsoid_bm((0, 0, 0), (r * 0.16,) * 3, 1.0, 12, 8), [hl])
                g.location = out * r * 0.93 + Vector((0, -s * r * 0.15, r * 0.35))
                self._attach(g, piv)

    def _visor(self):
        """Neon headband around the crown with a small cyan visor bar."""
        frame = principled("visor_frame", hexrgb("#14121f"), rough=0.25, Metallic=0.8)
        pts = []
        for i in range(41):
            a = -math.pi * 0.62 + 1.24 * math.pi * i / 40
            pts.append(Vector((
                0.12 + math.cos(a) * CEPH_R[0] * 0.92,
                math.sin(a) * CEPH_R[1] * 1.01,
                0.5 - 0.12 * math.cos(a),
            )))
        for name, z_off, mat, rad in (("band", 0.0, self.magenta, 0.045), ("band_frame", -0.07, frame, 0.07)):
            cu = bpy.data.curves.new(name, "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = rad
            sp = cu.splines.new("POLY")
            sp.points.add(len(pts) - 1)
            for p, v in zip(sp.points, pts):
                p.co = (v.x, v.y, v.z + z_off, 1)
            cu.materials.append(mat)
            self._attach(link(bpy.data.objects.new(name, cu)))
        visor = mesh_obj("visor", superellipsoid_bm((0.62, 0, 0.62), (0.16, 0.42, 0.1), 0.35), [frame])
        self._attach(visor).rotation_euler = (0, -0.5, 0)
        lens = mesh_obj("visor_lens", superellipsoid_bm((0.72, 0, 0.66), (0.08, 0.36, 0.05), 0.3), [self.cyan])
        self._attach(lens).rotation_euler = (0, -0.5, 0)

    # ---------------------------------------------------------- legs
    def _leg(self, i, side):
        rng = self.rng
        (hx, hy), lens, w = LEGS[i]
        k_ = C["fur_dk"]
        stops = [(0.0, k_), (0.5, C["fur"]), (1.0, C["fur_hi"])]
        segs = []
        r = w
        for si, ln in enumerate(lens):
            r1 = r * (0.85 if si < 2 else 0.6)
            ob = mesh_obj(f"leg{i}{si}", cone_bm(ln, r, r1), [band_ramp_material(f"leg{i}{si}", stops)])
            k = int(1400 * ln)
            p, n, t = cone_samples(rng, k, ln, r, r1)
            fur = strands(f"legfur{i}{si}", p, n, (0, 0, 1), rng.uniform(0.03, 0.06, k),
                          _fur_cols(rng, k), self.hair_m, rng, npts=4, bend=0.6,
                          r_root=0.007, r_tip=0.001)
            fur.parent = ob
            if si > 0:
                # glowing joint ring (knee / ankle)
                jr = mesh_obj(f"joint{i}{si}", torus_bm(r * 1.15, r * 0.22, 24, 8),
                              [self.cyan if si == 1 else self.magenta])
                jr.parent = ob
            segs.append(ob)
            r = r1
        return {"i": i, "side": side, "segs": segs, "lens": lens,
                "hip": Vector((hx, side * hy, HIP_Z))}

    # ---------------------------------------------------------- posing
    @staticmethod
    def _seg_matrix(a, b, pole):
        z = (b - a).normalized()
        x = pole - pole.project(z)
        if x.length < 1e-6:
            x = Vector((1, 0, 0))
        x.normalize()
        y = z.cross(x)
        m = Matrix((x, y, z)).transposed().to_4x4()
        m.translation = a
        return m

    def pose(self, M, feet_world):
        """M: spider world matrix. feet_world: 8 foot targets ordered like legs."""
        self.root.matrix_world = M
        up = (M.to_3x3() @ Vector((0, 0, 1))).normalized()
        sc = M.to_scale().x
        for leg, foot in zip(self.legs, feet_world):
            l1, l2, l3 = (ln * sc for ln in leg["lens"])
            hip = M @ leg["hip"]
            out = foot - hip
            out_h = out - out.project(up)
            out_h = out_h.normalized() if out_h.length > 1e-6 else Vector((1, 0, 0))
            ankle = foot + (out_h * 0.25 + up * 0.97).normalized() * l3
            d = ankle - hip
            dist = min(max(d.length, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
            dn = d.normalized()
            ankle = hip + dn * dist
            a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
            h = math.sqrt(max(0.0, l1 * l1 - a * a))
            pole = (up - out_h * 0.15) - (up - out_h * 0.15).project(dn)
            pole = pole.normalized() if pole.length > 1e-6 else up
            knee = hip + dn * a + pole * h
            pts = [hip, knee, ankle, foot]
            for k, ob in enumerate(leg["segs"]):
                pa, pb = pts[k], pts[k + 1]
                m = self._seg_matrix(pa, pb, out_h)
                zs = (pb - pa).length / l3 if k == 2 else 1.0
                ob.matrix_world = m @ Matrix.Diagonal((sc, sc, sc * zs, 1))

    def blink(self, amount):
        """amount 0 = open, 1 = shut. Squashes the eyes on their local Z."""
        for piv in self.eye_pivots:
            piv.scale = (1, 1, max(0.08, 1 - amount))
