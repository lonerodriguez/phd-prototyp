#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-konforme Stoßstellenanalyse (Hellwig Diss-Auszug) – XY-rotationsinvariant

Ziel:
- Egal ob IFC im XY um 90° gedreht ist (x/y vertauscht): gleiche Junction-Ergebnisse.
- Keine semantische Abhängigkeit von global X/Y, sondern lokale Achsen pro Element:
  - Wall: long_axis (Länge), thin_axis (Dicke/Normal)
  - Slab: axis=z, plus long_axis/short_axis in XY

Beibehaltende Fixes:
- clamp(FE.center) Kontaktpunkt auf SE-Face
- compute_dd Tie-Breaking: bei Wall–Slab Z-Faces priorisieren
- Regel Xh1-24-3 enthalten
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Set, Tuple

import ifcopenshell
import ifcopenshell.geom
import numpy as np

# -----------------------------
# Paper / heuristic constants
# -----------------------------
OFF_03 = 0.30
OFF_05 = 0.50
CLOSE_TO = 0.30

MIN_OVERLAP_LEN = 0.05
MIN_OVERLAP_AREA = 0.01

EDGE_GAP_TOL = 0.02
DD_TIE_EPS = 1e-9

Vec3 = Tuple[float, float, float]
DD = str  # "Xplus","Xminus","Yplus","Yminus","Zplus","Zminus"

# -----------------------------
# Geometry primitives
# -----------------------------
@dataclass(frozen=True)
class BBox:
    mn: Vec3
    mx: Vec3

    def distance_to(self, other: "BBox") -> float:
        dx = max(0.0, other.mn[0] - self.mx[0], self.mn[0] - other.mx[0])
        dy = max(0.0, other.mn[1] - self.mx[1], self.mn[1] - other.mx[1])
        dz = max(0.0, other.mn[2] - self.mx[2], self.mn[2] - other.mx[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def clamp_point(self, p: Vec3) -> Vec3:
        return (
            min(max(p[0], self.mn[0]), self.mx[0]),
            min(max(p[1], self.mn[1]), self.mx[1]),
            min(max(p[2], self.mn[2]), self.mx[2]),
        )

    def center(self) -> Vec3:
        return (
            (self.mn[0] + self.mx[0]) * 0.5,
            (self.mn[1] + self.mx[1]) * 0.5,
            (self.mn[2] + self.mx[2]) * 0.5,
        )

    def size(self) -> Vec3:
        return (self.mx[0] - self.mn[0], self.mx[1] - self.mn[1], self.mx[2] - self.mn[2])


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def gap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def overlap_lengths(a: BBox, b: BBox) -> Tuple[float, float, float]:
    ox = overlap_1d(a.mn[0], a.mx[0], b.mn[0], b.mx[0])
    oy = overlap_1d(a.mn[1], a.mx[1], b.mn[1], b.mx[1])
    oz = overlap_1d(a.mn[2], a.mx[2], b.mn[2], b.mx[2])
    return ox, oy, oz


def xy_gaps(a: BBox, b: BBox) -> Tuple[float, float]:
    gx = gap_1d(a.mn[0], a.mx[0], b.mn[0], b.mx[0])
    gy = gap_1d(a.mn[1], a.mx[1], b.mn[1], b.mx[1])
    return gx, gy


def z_gap(a: BBox, b: BBox) -> float:
    return gap_1d(a.mn[2], a.mx[2], b.mn[2], b.mx[2])


# -----------------------------
# IFC element data
# -----------------------------
@dataclass
class ElementInfo:
    ifc_id: int
    guid: str
    ifc_type: str
    name: str
    bbox: BBox

    # paper-axis:
    # - wall: thin_axis (Normal/Dicke)
    # - slab: "z"
    axis: str

    # local horizontal axes (rotation invariant):
    # - wall: long_axis (length dir), thin_axis (normal/thickness dir)
    # - slab: long_axis (bigger XY), short_axis (smaller XY)
    long_axis: Optional[str] = None
    short_axis: Optional[str] = None

    # for wall convenience
    thin_axis: Optional[str] = None

    # dir-labels: n/m/o
    dir_label: str = ""


@dataclass
class JB:
    jb_id: int
    bbox: BBox
    se_id: int
    fe_ids: List[int]


# -----------------------------
# IFC geometry
# -----------------------------
def create_geom_settings():
    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, True)
    return s


def element_bbox(settings, element) -> Optional[BBox]:
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape((-1, 3))
        mn = tuple(np.min(verts, axis=0).tolist())
        mx = tuple(np.max(verts, axis=0).tolist())
        return BBox(mn, mx)
    except Exception:
        return None


def is_wall(t: str) -> bool:
    return t.lower().startswith("ifcwall")


def is_slab(t: str) -> bool:
    tl = t.lower()
    return tl.startswith("ifcslab") or tl.startswith("ifcroof")


def collect_walls_slabs(model) -> List:
    els = []
    els.extend(model.by_type("IfcWall"))
    els.extend(model.by_type("IfcWallStandardCase"))
    els.extend(model.by_type("IfcSlab"))
    els.extend(model.by_type("IfcRoof"))
    out, seen = [], set()
    for e in els:
        if e.id() not in seen:
            out.append(e)
            seen.add(e.id())
    return out


def classify_local_axes(ifc_type: str, b: BBox) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Rotation-invariant in XY:
    - wall: long_axis = max(sx,sy), thin_axis = min(sx,sy), axis(thin) = thin_axis
    - slab: axis = 'z', long_axis/short_axis in XY
    Returns: (axis, long_axis, short_axis, thin_axis)
    """
    sx, sy, _ = b.size()

    if is_slab(ifc_type):
        long_axis = "x" if sx >= sy else "y"
        short_axis = "y" if long_axis == "x" else "x"
        return "z", long_axis, short_axis, None

    # wall
    long_axis = "x" if sx >= sy else "y"
    thin_axis = "y" if long_axis == "x" else "x"
    # keep short_axis same as thin_axis for walls if you want
    return thin_axis, long_axis, thin_axis, thin_axis


def wall_length_axis(w: ElementInfo) -> str:
    # w.axis ist Dickenrichtung ("x" oder "y")
    # Länge ist die andere horizontale Achse
    return "y" if w.axis == "x" else "x"


def is_near_wall_end(w: ElementInfo, p_on_w: Vec3) -> bool:
    la = wall_length_axis(w)
    v = p_on_w[0] if la == "x" else p_on_w[1]
    mn = w.bbox.mn[0] if la == "x" else w.bbox.mn[1]
    mx = w.bbox.mx[0] if la == "x" else w.bbox.mx[1]
    tol = OFF_05 + EDGE_GAP_TOL + 1e-9
    return (v - mn) <= tol or (mx - v) <= tol
# -----------------------------
# Candidate/contact filter
# -----------------------------
def has_sufficient_contact(fe: ElementInfo, se: ElementInfo) -> bool:
    if se.bbox.distance_to(fe.bbox) > CLOSE_TO:
        return False

    ox, oy, oz = overlap_lengths(se.bbox, fe.bbox)
    gx, gy = xy_gaps(se.bbox, fe.bbox)
    zg = z_gap(se.bbox, fe.bbox)

    se_is_slab = is_slab(se.ifc_type)
    fe_is_slab = is_slab(fe.ifc_type)
    se_is_wall = is_wall(se.ifc_type)
    fe_is_wall = is_wall(fe.ifc_type)

    # Wall–Slab
    if (se_is_wall and fe_is_slab) or (se_is_slab and fe_is_wall):
        area_xy = ox * oy
        if (ox >= MIN_OVERLAP_LEN and oy >= MIN_OVERLAP_LEN and area_xy >= MIN_OVERLAP_AREA):
            return True

        if oz >= MIN_OVERLAP_LEN:
            if ox >= MIN_OVERLAP_LEN and gy <= EDGE_GAP_TOL:
                return True
            if oy >= MIN_OVERLAP_LEN and gx <= EDGE_GAP_TOL:
                return True
        return False

    # Slab–Slab
    if se_is_slab and fe_is_slab:
        area_xy = ox * oy
        return (ox >= MIN_OVERLAP_LEN and oy >= MIN_OVERLAP_LEN and area_xy >= MIN_OVERLAP_AREA)

    # Wall–Wall
    if se_is_wall and fe_is_wall:
        if oz >= MIN_OVERLAP_LEN and ((ox >= MIN_OVERLAP_LEN) or (oy >= MIN_OVERLAP_LEN)):
            return True

        area_xy = ox * oy
        if zg <= EDGE_GAP_TOL and ox >= MIN_OVERLAP_LEN and oy >= MIN_OVERLAP_LEN and area_xy >= MIN_OVERLAP_AREA:
            return True

        return False

    return True


# -----------------------------
# Junction Boxes (JB1..JB6) – rotationsinvariant in XY
# -----------------------------
def build_junction_boxes(se: ElementInfo) -> Dict[int, JB]:
    mnx, mny, mnz = se.bbox.mn
    mxx, mxy, mxz = se.bbox.mx

    def mk(jb_id: int, mn: Vec3, mx: Vec3) -> JB:
        return JB(jb_id=jb_id, bbox=BBox(mn, mx), se_id=se.ifc_id, fe_ids=[])

    jbs: Dict[int, JB] = {}

    # ---- slab: split by local long/short (not hard x/y)
    if is_slab(se.ifc_type):
        la = se.long_axis or "x"
        sa = se.short_axis or ("y" if la == "x" else "x")

        def get_min(ax: str) -> float:
            return mnx if ax == "x" else mny

        def get_max(ax: str) -> float:
            return mxx if ax == "x" else mxy

        # JB1/2/3 along long-axis
        # JB4/5/6 along short-axis
        # z expanded by OFF_03 like before
        la_min, la_max = get_min(la), get_max(la)
        sa_min, sa_max = get_min(sa), get_max(sa)

        # helper to build bbox with axis-aligned coords
        def slab_box(la_rng: Tuple[float, float], sa_rng: Tuple[float, float]) -> Tuple[Vec3, Vec3]:
            # map ranges into x/y
            if la == "x":
                x0, x1 = la_rng
                y0, y1 = sa_rng
            else:
                y0, y1 = la_rng
                x0, x1 = sa_rng
            return (x0, y0, mnz - OFF_03), (x1, y1, mxz + OFF_03)

        # Along long-axis strips (JB1..3): short-axis full, long-axis banded
        jb1_la = (la_min - OFF_05, la_min + OFF_05)
        jb2_la = (la_min + OFF_05, la_max - OFF_05)
        jb3_la = (la_max - OFF_05, la_max + OFF_05)
        full_sa = (sa_min - OFF_05, sa_max + OFF_05)

        mn, mx = slab_box(jb1_la, full_sa); jbs[1] = mk(1, mn, mx)
        mn, mx = slab_box(jb2_la, full_sa); jbs[2] = mk(2, mn, mx)
        mn, mx = slab_box(jb3_la, full_sa); jbs[3] = mk(3, mn, mx)

        # Along short-axis strips (JB4..6): long-axis full, short-axis banded
        jb4_sa = (sa_min - OFF_05, sa_min + OFF_05)
        jb5_sa = (sa_min + OFF_05, sa_max - OFF_05)
        jb6_sa = (sa_max - OFF_05, sa_max + OFF_05)
        full_la = (la_min - OFF_05, la_max + OFF_05)

        mn, mx = slab_box(full_la, jb4_sa); jbs[4] = mk(4, mn, mx)
        mn, mx = slab_box(full_la, jb5_sa); jbs[5] = mk(5, mn, mx)
        mn, mx = slab_box(full_la, jb6_sa); jbs[6] = mk(6, mn, mx)

        return jbs

    # ---- wall: use thin_axis + long_axis (rotation invariant)
    ta = se.thin_axis or se.axis          # thickness/normal axis in XY
    la = se.long_axis or ("y" if ta == "x" else "x")  # length axis in XY

    def get_min(ax: str) -> float:
        return mnx if ax == "x" else mny

    def get_max(ax: str) -> float:
        return mxx if ax == "x" else mxy

    ta_min, ta_max = get_min(ta), get_max(ta)
    la_min, la_max = get_min(la), get_max(la)

    # helper: create a wall JB by giving ranges in ta & la
    def wall_box(ta_rng: Tuple[float, float], la_rng: Tuple[float, float], z_rng: Tuple[float, float]) -> Tuple[Vec3, Vec3]:
        if ta == "x":
            x0, x1 = ta_rng
            y0, y1 = la_rng
        else:
            y0, y1 = ta_rng
            x0, x1 = la_rng
        z0, z1 = z_rng
        return (x0, y0, z0), (x1, y1, z1)

    # JB1/2/3: split along wall length axis (la)
    ta_rng_small = (ta_min - OFF_03, ta_max + OFF_03)
    jb1_la = (la_min - OFF_05, la_min + OFF_05)
    jb2_la = (la_min + OFF_05, la_max - OFF_05)
    jb3_la = (la_max - OFF_05, la_max + OFF_05)

    mn, mx = wall_box(ta_rng_small, jb1_la, (mnz, mxz)); jbs[1] = mk(1, mn, mx)
    mn, mx = wall_box(ta_rng_small, jb2_la, (mnz, mxz)); jbs[2] = mk(2, mn, mx)
    mn, mx = wall_box(ta_rng_small, jb3_la, (mnz, mxz)); jbs[3] = mk(3, mn, mx)

    # JB4/5/6: split along Z, cover full length (la) with OFF_05, thickness with OFF_03
    la_rng_full = (la_min - OFF_05, la_max + OFF_05)
    jb4_z = (mnz - OFF_03, mnz + OFF_03)
    jb5_z = (mnz + OFF_03, mxz - OFF_03)
    jb6_z = (mxz - OFF_03, mxz + OFF_03)

    mn, mx = wall_box(ta_rng_small, la_rng_full, jb4_z); jbs[4] = mk(4, mn, mx)
    mn, mx = wall_box(ta_rng_small, la_rng_full, jb5_z); jbs[5] = mk(5, mn, mx)
    mn, mx = wall_box(ta_rng_small, la_rng_full, jb6_z); jbs[6] = mk(6, mn, mx)

    return jbs


# -----------------------------
# DD + FE->JB assignment (tie-breaking stays)
# -----------------------------
def dd_axis_and_sign(dd: str) -> Tuple[str, str]:
    # returns ("x"|"y"|"z", "plus"|"minus")
    if dd.startswith("X"):
        return "x", "plus" if dd.endswith("plus") else "minus"
    if dd.startswith("Y"):
        return "y", "plus" if dd.endswith("plus") else "minus"
    return "z", "plus" if dd.endswith("plus") else "minus"


def compute_dd(fe: ElementInfo, se: ElementInfo) -> DD:
    """
    Robust DD: nearest face on SE to FE.
    FIX: Tie-break:
      - if se is Wall and fe is Slab -> prefer Z faces (Lv1-2 fix)
      - if se is Slab and fe is Wall -> prefer Z only if strong XY overlap, else prefer X/Y (Tv1-24 vs Tv2-13)
    """
    p = se.bbox.clamp_point(fe.bbox.center())
    smn, smx = se.bbox.mn, se.bbox.mx

    face_dists = [
        ("Xminus", abs(p[0] - smn[0])),
        ("Xplus",  abs(smx[0] - p[0])),
        ("Yminus", abs(p[1] - smn[1])),
        ("Yplus",  abs(smx[1] - p[1])),
        ("Zminus", abs(p[2] - smn[2])),
        ("Zplus",  abs(smx[2] - p[2])),
    ]
    minv = min(d for _, d in face_dists)
    candidates = [k for k, d in face_dists if abs(d - minv) <= DD_TIE_EPS]

    if len(candidates) == 1:
        return candidates[0]

    # --- tie-break priorities ---
    if is_wall(se.ifc_type) and is_slab(fe.ifc_type):
        # original fix for Lv1-2
        pref = ["Zplus", "Zminus", "Yplus", "Yminus", "Xplus", "Xminus"]

    elif is_slab(se.ifc_type) and is_wall(fe.ifc_type):
        # decide by overlap pattern (Tv1-24 vs Tv2-13)
        ox, oy, _ = overlap_lengths(se.bbox, fe.bbox)
        strong_xy = (ox >= MIN_OVERLAP_LEN and oy >= MIN_OVERLAP_LEN)

        if strong_xy:
            # wall overlaps slab in plan -> treat as top/bottom contact
            pref = ["Zplus", "Zminus", "Yplus", "Yminus", "Xplus", "Xminus"]
        else:
            # edge contact -> use slab side face (stirnfläche)
            pref = ["Yplus", "Yminus", "Xplus", "Xminus", "Zplus", "Zminus"]

    else:
        pref = ["Xplus", "Xminus", "Yplus", "Yminus", "Zplus", "Zminus"]

    for k in pref:
        if k in candidates:
            return k

    return candidates[0]


def assign_fe_to_jb(fe: ElementInfo, se: ElementInfo, jbs: Dict[int, JB]) -> Optional[int]:
    """
    Rotation invariant:
    - For walls, interpret se.thin_axis (ta) and se.long_axis (la)
    - Map dd to axis+sign and apply the original intent without hardcoding x/y cases
    """
    dd = compute_dd(fe, se)
    dd_ax, dd_sign = dd_axis_and_sign(dd)

    FE = fe.bbox
    JB1 = jbs[1].bbox
    JB3 = jbs[3].bbox
    JB4 = jbs[4].bbox
    JB6 = jbs[6].bbox

    # slab SE: keep logic but now JBs already built rotation invariant -> we can use dd axis directly
    if is_slab(se.ifc_type):
        # For slab, use: dd in x/y -> which side strip; dd in z -> decide by FE position in-plane.
        if dd_ax == "x":
            return 6 if dd_sign == "plus" else 4
        if dd_ax == "y":
            return 3 if dd_sign == "plus" else 1
        # dd_ax == "z": pick based on FE position along slab-long axis
        # choose JB3/JB1/JB2 by where FE lies relative to JB bands
        # We use the slab's long_axis and compare that coordinate.
        la = se.long_axis or "x"

        def fe_min(ax: str) -> float:
            return FE.mn[0] if ax == "x" else FE.mn[1]

        def fe_max(ax: str) -> float:
            return FE.mx[0] if ax == "x" else FE.mx[1]

        def jb_min(b: BBox, ax: str) -> float:
            return b.mn[0] if ax == "x" else b.mn[1]

        def jb_max(b: BBox, ax: str) -> float:
            return b.mx[0] if ax == "x" else b.mx[1]

        if fe_min(la) >= jb_min(JB3, la):
            return 3
        if fe_max(la) <= jb_max(JB1, la):
            return 1
        return 2

    # wall SE:
    ta = se.thin_axis or se.axis
    la = se.long_axis or ("y" if ta == "x" else "x")

    fe_n = fe.axis  # wall: thin_axis, slab: z

    def fe_min(ax: str) -> float:
        return FE.mn[0] if ax == "x" else FE.mn[1] if ax == "y" else FE.mn[2]

    def fe_max(ax: str) -> float:
        return FE.mx[0] if ax == "x" else FE.mx[1] if ax == "y" else FE.mx[2]

    def jb_min(b: BBox, ax: str) -> float:
        return b.mn[0] if ax == "x" else b.mn[1] if ax == "y" else b.mn[2]

    def jb_max(b: BBox, ax: str) -> float:
        return b.mx[0] if ax == "x" else b.mx[1] if ax == "y" else b.mx[2]

    # Case 1: FE is slab (fe_n == 'z') contacting wall
    if fe_n == "z":
        if dd_ax == ta:
            # choose JB6/JB4/JB5 by FE z position
            if fe_min("z") >= jb_min(JB6, "z"):
                return 6
            if fe_max("z") <= jb_max(JB4, "z"):
                return 4
            return 5
        if dd_ax == "z":
            return 6 if dd_sign == "plus" else 4
        # otherwise undefined
        return None

    # Case 2: FE wall parallel to SE wall (same thin axis)
    if fe_n == ta:
        if dd_ax == la:
            return 3 if dd_sign == "plus" else 1
        if dd_ax == "z":
            return 6 if dd_sign == "plus" else 4
        return None

    # Case 3: FE wall perpendicular (thin axis equals SE long axis)
    if fe_n == la:
        if dd_ax == ta:
            # choose 3/1/2 by where FE lies along SE length axis
            if fe_min(la) >= jb_min(JB3, la):
                return 3
            if fe_max(la) <= jb_max(JB1, la):
                return 1
            return 2
        if dd_ax == la:
            return 3 if dd_sign == "plus" else 1
        return None

    return None


# -----------------------------
# Connection zones (contact point fix stays)
# -----------------------------
def face_areas_from_bbox(b: BBox) -> Dict[str, float]:
    sx, sy, sz = b.size()
    return {"x": sy * sz, "y": sx * sz, "z": sx * sy}


def smallest_face_axes(b: BBox) -> Set[str]:
    areas = face_areas_from_bbox(b)
    sorted_axes = sorted(areas.keys(), key=lambda k: areas[k])
    return {sorted_axes[0], sorted_axes[1]}


def point_on_dd_face(ei: ElementInfo, ej: ElementInfo) -> Tuple[Vec3, str]:
    bi = ei.bbox
    dd = compute_dd(ej, ei)
    base = bi.clamp_point(ej.bbox.center())
    x, y, z = base

    if dd == "Xminus":
        x = bi.mn[0]; face_axis = "x"
    elif dd == "Xplus":
        x = bi.mx[0]; face_axis = "x"
    elif dd == "Yminus":
        y = bi.mn[1]; face_axis = "y"
    elif dd == "Yplus":
        y = bi.mx[1]; face_axis = "y"
    elif dd == "Zminus":
        z = bi.mn[2]; face_axis = "z"
    else:
        z = bi.mx[2]; face_axis = "z"

    return (x, y, z), face_axis


def border_strip_by_distance(elem: ElementInfo, p_on_elem: Vec3, large_face_axis: str) -> bool:
    b = elem.bbox
    in_plane = [ax for ax in ("x", "y", "z") if ax != large_face_axis]

    def coord(ax: str) -> float:
        return p_on_elem[0] if ax == "x" else p_on_elem[1] if ax == "y" else p_on_elem[2]

    for ax in in_plane:
        v = coord(ax)
        mn = b.mn[0] if ax == "x" else b.mn[1] if ax == "y" else b.mn[2]
        mx = b.mx[0] if ax == "x" else b.mx[1] if ax == "y" else b.mx[2]
        if (v - mn) <= OFF_05 + 1e-9 or (mx - v) <= OFF_05 + 1e-9:
            return True
    return False


def border_by_jb_paper(se: ElementInfo, fe_axis: str, fe_jb: Optional[int]) -> Optional[bool]:
    """
    Rotation invariant:
    - For wall SE:
        * FE along SE long-axis => border if JB in (1,3)
        * FE along z           => border if JB in (4,6)
    - For slab SE:
        * FE along slab long_axis  => border if JB in (1,3)
        * FE along slab short_axis => border if JB in (4,6)
    """
    if fe_jb is None:
        return None

    if is_wall(se.ifc_type):
        la = se.long_axis
        if la is None:
            # fallback
            la = "y" if (se.axis == "x") else "x"
        if fe_axis == la:
            return fe_jb in (1, 3)
        if fe_axis == "z":
            return fe_jb in (4, 6)
        return None

    if is_slab(se.ifc_type):
        la = se.long_axis or "x"
        sa = se.short_axis or ("y" if la == "x" else "x")
        if fe_axis == la:
            return fe_jb in (1, 3)
        if fe_axis == sa:
            return fe_jb in (4, 6)
        return None

    return None


def cz_for_pair(ei: ElementInfo, ej: ElementInfo, ej_jb_if_ei_is_se: Optional[int]) -> str:
    p, face_axis = point_on_dd_face(ei, ej)

    small_axes = smallest_face_axes(ei.bbox)
    if face_axis in small_axes:
        return "short"

    # --- FIX: Wall–Wall Border/Middle über Wand-Ende statt JB ---
    if is_wall(ei.ifc_type) and is_wall(ej.ifc_type):
        return "border" if is_near_wall_end(ei, p) else "middle"

    jb_border = border_by_jb_paper(ei, ej.axis, ej_jb_if_ei_is_se)
    if jb_border is not None:
        return "border" if jb_border else "middle"

    return "border" if border_strip_by_distance(ei, p, face_axis) else "middle"


def assign_dir_labels(elements: List[ElementInfo]) -> None:
    # slabs are always "o"
    for e in elements:
        if is_slab(e.ifc_type):
            e.dir_label = "o"

    walls = [e for e in elements if is_wall(e.ifc_type)]
    if not walls:
        return

    # use wall ORIENTATION = long_axis (rotation invariant), not global axis name meaning
    ref = walls[0]
    ref.dir_label = "n"
    ref_or = ref.long_axis

    for w in walls[1:]:
        w.dir_label = "n" if w.long_axis == ref_or else "m"


def build_cz_matrix(elements: List[ElementInfo], se_id: int, fe_to_jb: Dict[int, Optional[int]]) -> Dict[int, Dict[int, str]]:
    by_id = {e.ifc_id: e for e in elements}
    ids = [e.ifc_id for e in elements]
    cz: Dict[int, Dict[int, str]] = {i: {} for i in ids}

    for i in ids:
        for j in ids:
            if i == j:
                continue
            ei, ej = by_id[i], by_id[j]
            if not has_sufficient_contact(ej, ei):
                continue
            ej_jb = fe_to_jb.get(j) if i == se_id else None
            cz[i][j] = cz_for_pair(ei, ej, ej_jb)

    se = by_id[se_id]
    fl = [i for i in ids if i != se_id]
    for a, b in combinations(fl, 2):
        ea, eb = by_id[a], by_id[b]
        if has_sufficient_contact(eb, ea):
            continue
        if ea.bbox.distance_to(se.bbox) <= CLOSE_TO and eb.bbox.distance_to(se.bbox) <= CLOSE_TO:
            cz[a][b] = "0"
            cz[b][a] = "0"

    return cz


# -----------------------------
# Junction type rules
# -----------------------------
def mk_rule(jtype: str, dirs: List[str], cz_constraints: List[Tuple[int, int, str]]) -> Dict[str, Any]:
    return {"type": jtype, "k": len(dirs), "dirs": dirs, "cz": cz_constraints}


RULES: List[Dict[str, Any]] = [
    mk_rule("Lh1-2", ["n", "m"], [(1, 2, "short"), (2, 1, "border")]),
    mk_rule("Lv1-2", ["n", "o"], [(1, 2, "short"), (2, 1, "border")]),

    mk_rule("Tv2-13", ["n", "o"], [(1, 2, "short"), (2, 1, "middle")]),
    mk_rule("Th1-24", ["n", "m"], [(1, 2, "short"), (2, 1, "middle")]),
    mk_rule("Tv1-24", ["n", "o"], [(1, 2, "middle"), (2, 1, "short")]),

    mk_rule("Th2-1-4", ["n", "m", "m"], [
        (1, 2, "border"), (1, 3, "border"),
        (2, 1, "short"),  (3, 1, "short"),
        (2, 3, "0"),      (3, 2, "0"),
    ]),

    mk_rule("Xh1-24-3", ["n", "m", "m"], [
        (1, 2, "middle"), (1, 3, "middle"),
        (2, 1, "short"),  (3, 1, "short"),
        (2, 3, "0"),      (3, 2, "0"),
    ]),

    # Tv2-1-3 (Wall + 2 slabs)
    mk_rule("Tv2-1-3", ["n", "o", "o"], [
        (1, 2, "short"), (1, 3, "short"),
        (2, 1, "border"), (3, 1, "border"),
        (2, 3, "0"), (3, 2, "0"),
    ]),
]


def multiset(lst: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


def match_rule(elements: List[ElementInfo], se_id: int, fe_to_jb: Dict[int, Optional[int]]) -> Tuple[str, Dict[str, Any]]:
    assign_dir_labels(elements)
    cz = build_cz_matrix(elements, se_id, fe_to_jb)

    dbg = {
        "k": len(elements),
        "dirs": {e.ifc_id: e.dir_label for e in elements},
        "cz_pairs": {str(i): cz[i] for i in cz},
        "se_id": se_id,
        "fe_to_jb": fe_to_jb,
    }

    ids = [e.ifc_id for e in elements]
    id_to_dir = {e.ifc_id: e.dir_label for e in elements}

    for rule in RULES:
        if rule["k"] != len(elements):
            continue
        if multiset(rule["dirs"]) != multiset([id_to_dir[i] for i in ids]):
            continue

        role_dirs = {r + 1: rule["dirs"][r] for r in range(rule["k"])}

        for perm in permutations(ids):
            role_to_id = {r + 1: perm[r] for r in range(len(perm))}

            ok = True
            for r, reqd in role_dirs.items():
                if id_to_dir[role_to_id[r]] != reqd:
                    ok = False
                    break
            if not ok:
                continue

            for (ra, rb, want) in rule["cz"]:
                a_id = role_to_id[ra]
                b_id = role_to_id[rb]
                got = cz.get(a_id, {}).get(b_id)
                if got != want:
                    ok = False
                    break

            if ok:
                dbg["matched_rule_perm"] = {str(r): role_to_id[r] for r in role_to_id}
                return rule["type"], dbg

    return "UNKNOWN", dbg


# -----------------------------
# Post processing
# -----------------------------
def junction_score(j: Dict[str, Any]) -> Tuple[int, int]:
    jt = j.get("junction_type", "UNKNOWN")
    recognized = 1 if jt not in ("UNKNOWN", "NONE") else 0
    size = len(j.get("element_ids", []))
    return (recognized, size)


def keep_only_maximal_junctions(junctions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sets = [(frozenset(j["element_ids"]), j) for j in junctions]
    sets.sort(key=lambda x: junction_score(x[1]), reverse=True)

    kept: List[Tuple[frozenset, Dict[str, Any]]] = []
    for s, j in sets:
        if any(s < ks for ks, _ in kept):
            continue
        kept.append((s, j))

    out = [j for _, j in kept]
    out.sort(key=lambda r: (r["junction_type"], r["element_ids"]))
    return out


# -----------------------------
# Analyze IFC
# -----------------------------
def analyze(ifc_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = ifcopenshell.open(ifc_path)
    settings = create_geom_settings()

    raw = collect_walls_slabs(model)

    infos: Dict[int, ElementInfo] = {}
    for e in raw:
        bb = element_bbox(settings, e)
        if bb is None:
            continue

        axis, long_axis, short_axis, thin_axis = classify_local_axes(e.is_a(), bb)

        infos[e.id()] = ElementInfo(
            ifc_id=e.id(),
            guid=getattr(e, "GlobalId", "") or "",
            ifc_type=e.is_a(),
            name=getattr(e, "Name", "") or "",
            bbox=bb,
            axis=axis,
            long_axis=long_axis,
            short_axis=short_axis,
            thin_axis=thin_axis,
        )

    se_ids = [i for i, inf in infos.items() if is_wall(inf.ifc_type) or is_slab(inf.ifc_type)]
    rows: List[Dict[str, Any]] = []

    for se_id in se_ids:
        se = infos[se_id]
        jbs = build_junction_boxes(se)

        candidates: List[ElementInfo] = []
        for fe_id, fe in infos.items():
            if fe_id == se_id:
                continue
            if not (is_wall(fe.ifc_type) or is_slab(fe.ifc_type)):
                continue
            if has_sufficient_contact(fe, se):
                candidates.append(fe)

        fe_to_jb_all: Dict[int, Optional[int]] = {}
        for fe in candidates:
            jb_id = assign_fe_to_jb(fe, se, jbs)
            fe_to_jb_all[fe.ifc_id] = jb_id
            if jb_id is None:
                continue
            if fe.ifc_id not in jbs[jb_id].fe_ids:
                jbs[jb_id].fe_ids.append(fe.ifc_id)

        for jb_id, jb in sorted(jbs.items(), key=lambda x: x[0]):
            fe_list = jb.fe_ids[:3]
            if not fe_list:
                continue
            elem_ids = [jb.se_id] + fe_list
            elems = [infos[i] for i in elem_ids]
            fe_to_jb = {fid: fe_to_jb_all.get(fid) for fid in fe_list}
            jtype, dbg = match_rule(elems, se_id, fe_to_jb)

            rows.append({
                "junction_scope": f"JB{jb_id}",
                "junction_box": jb_id,
                "element_ids": sorted(elem_ids),
                "junction_type": jtype,
                "elements": [{
                    "ifc_id": e.ifc_id,
                    "guid": e.guid,
                    "type": e.ifc_type,
                    "name": e.name,
                    "axis": e.axis,
                    "dir": e.dir_label,
                    "bbox": {"mn": list(e.bbox.mn), "mx": list(e.bbox.mx)},
                } for e in elems],
                "debug": dbg,
            })

        if candidates:
            cand_sorted = sorted(candidates, key=lambda fe: se.bbox.distance_to(fe.bbox))
            chosen = cand_sorted[:3]
            elem_ids = [se_id] + [fe.ifc_id for fe in chosen]
            elems = [infos[i] for i in elem_ids]
            fe_to_jb = {fe.ifc_id: fe_to_jb_all.get(fe.ifc_id) for fe in chosen}
            jtype, dbg = match_rule(elems, se_id, fe_to_jb)

            rows.append({
                "junction_scope": "COMBINED",
                "junction_box": None,
                "element_ids": sorted(elem_ids),
                "junction_type": jtype,
                "elements": [{
                    "ifc_id": e.ifc_id,
                    "guid": e.guid,
                    "type": e.ifc_type,
                    "name": e.name,
                    "axis": e.axis,
                    "dir": e.dir_label,
                    "bbox": {"mn": list(e.bbox.mn), "mx": list(e.bbox.mx)},
                } for e in elems],
                "debug": dbg,
            })

    best: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for r in rows:
        key = tuple(sorted(r["element_ids"]))
        if key not in best or junction_score(r) > junction_score(best[key]):
            best[key] = r

    unique = keep_only_maximal_junctions(list(best.values()))
    unique.sort(key=lambda x: (x["junction_type"], x["element_ids"]))
    return unique, rows


def main():
    ifc_path = "./ifc-models/Xh1-24-3.ifc" if len(sys.argv) < 2 else sys.argv[1]
    if not os.path.exists(ifc_path):
        print(f"ERROR: IFC not found: {ifc_path}")
        sys.exit(1)

    unique, debug = analyze(ifc_path)

    with open("junctions_unique.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    with open("junctions_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"- junctions_unique.json: {len(unique)}")
    print(f"- junctions_debug.json : {len(debug)}")


if __name__ == "__main__":
    main()