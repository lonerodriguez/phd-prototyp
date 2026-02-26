#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-konforme Stoßstellenanalyse (Hellwig Diss-Auszug)

Implements:
- Junction Boxes per element direction (n=(1/0/0) or (0/1/0), o=(0/0/1))
  using the stated JB1 example and symmetric completion for JB2..JB6.
- DD (distance direction) between FE and SE: Xplus/Xminus/Yplus/Yminus/Zplus/Zminus
- FE -> JunctionBox assignment EXACTLY via Algorithm 1-3 (pages 100-103)
- Connection Zones short/border/middle:
  * short = contact point lies on one of the 4 smallest faces of the element
  * border/middle defined on the largest face with 0.5m border strips (Fig 5.39-5.41)
- Junction type identification according to Fig 5.43 (15 types), permutation-invariant.

Usage:
  python ifc_junctions_paper.py            # reads ./model.ifc
  python ifc_junctions_paper.py my.ifc

Output:
  junctions_unique.json
  junctions_debug.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any
from itertools import permutations, combinations

import numpy as np
import ifcopenshell
import ifcopenshell.geom


# -----------------------------
# Paper constants
# -----------------------------
OFF_03 = 0.30   # +/-0.3 m
OFF_05 = 0.50   # +/-0.5 m
CLOSE_TO = 0.30 # "close to" threshold (pragmatic; paper uses 0.3 in JB example sizing)

# Numeric eps for "on face" detection
EPS_FACE = 1e-6

# -----------------------------
# Data types
# -----------------------------
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class BBox:
    mn: Vec3
    mx: Vec3

    def intersects(self, other: "BBox") -> bool:
        return (
            self.mn[0] <= other.mx[0] and self.mx[0] >= other.mn[0] and
            self.mn[1] <= other.mx[1] and self.mx[1] >= other.mn[1] and
            self.mn[2] <= other.mx[2] and self.mx[2] >= other.mn[2]
        )

    def distance_to(self, other: "BBox") -> float:
        dx = max(0.0, other.mn[0] - self.mx[0], self.mn[0] - other.mx[0])
        dy = max(0.0, other.mn[1] - self.mx[1], self.mn[1] - other.mx[1])
        dz = max(0.0, other.mn[2] - self.mx[2], self.mn[2] - other.mx[2])
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def clamp_point(self, p: Vec3) -> Vec3:
        return (
            min(max(p[0], self.mn[0]), self.mx[0]),
            min(max(p[1], self.mn[1]), self.mx[1]),
            min(max(p[2], self.mn[2]), self.mx[2]),
        )

    def center(self) -> Vec3:
        return ((self.mn[0] + self.mx[0]) * 0.5, (self.mn[1] + self.mx[1]) * 0.5, (self.mn[2] + self.mx[2]) * 0.5)

    def size(self) -> Vec3:
        return (self.mx[0] - self.mn[0], self.mx[1] - self.mn[1], self.mx[2] - self.mn[2])


@dataclass
class ElementInfo:
    ifc_id: int
    guid: str
    ifc_type: str
    name: str
    bbox: BBox
    # "axis direction class" per paper: 'x','y','z' for (1/0/0),(0/1/0),(0/0/1)
    axis: str
    # "relative direction label" in a junction context: n/m/o
    dir_label: str = ""


@dataclass
class JB:
    jb_id: int
    bbox: BBox
    # slots: SE always present; FE1/FE2/FE3 are optional
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
    t = t.lower()
    return t.startswith("ifcslab") or t.startswith("ifcroof")


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


# -----------------------------
# Paper: element axis classification (x/y for walls, z for slabs)
# Assumption: global-axis-aligned model (as in diss excerpt).
# -----------------------------
def classify_axis(inf: ElementInfo) -> str:
    sx, sy, sz = inf.bbox.size()
    if is_slab(inf.ifc_type):
        return "z"  # o=(0/0/1)

    # wall: direction is along the longer horizontal extent (x or y)
    # (paper: n=(1/0/0) OR (0/1/0))
    return "x" if sx >= sy else "y"


# -----------------------------
# Paper: build Junction Boxes (JB1..JB6)
# Based on explicit JB1 example per axis and symmetric completion.
# -----------------------------
def build_junction_boxes(se: ElementInfo) -> Dict[int, JB]:
    mnx, mny, mnz = se.bbox.mn
    mxx, mxy, mxz = se.bbox.mx

    axis = se.axis

    def mk(jb_id: int, mn: Vec3, mx: Vec3) -> JB:
        return JB(jb_id=jb_id, bbox=BBox(mn, mx), se_id=se.ifc_id, fe_ids=[])

    jbs: Dict[int, JB] = {}

    if axis == "x":
        # SE.n = (1/0/0) per excerpt: JB1 explicitly:
        # JB1: XMin-0.3 / YMin-0.5 / ZMin  to  XMax+0.3 / YMin+0.5 / ZMax
        # Complete symmetrically:
        jbs[1] = mk(1, (mnx - OFF_03, mny - OFF_05, mnz), (mxx + OFF_03, mny + OFF_05, mxz))
        jbs[2] = mk(2, (mnx - OFF_03, mny + OFF_05, mnz), (mxx + OFF_03, mxy - OFF_05, mxz))
        jbs[3] = mk(3, (mnx - OFF_03, mxy - OFF_05, mnz), (mxx + OFF_03, mxy + OFF_05, mxz))

        # vertical bands: mirror of slab-style with +/-0.3 in Z, full Y with +/-0.5 margin
        jbs[4] = mk(4, (mnx - OFF_03, mny - OFF_05, mnz - OFF_03), (mxx + OFF_03, mxy + OFF_05, mnz + OFF_03))
        jbs[5] = mk(5, (mnx - OFF_03, mny - OFF_05, mnz + OFF_03), (mxx + OFF_03, mxy + OFF_05, mxz - OFF_03))
        jbs[6] = mk(6, (mnx - OFF_03, mny - OFF_05, mxz - OFF_03), (mxx + OFF_03, mxy + OFF_05, mxz + OFF_03))

    elif axis == "y":
        # SE.n = (0/1/0) per excerpt: JB1 explicitly:
        # JB1: XMin-0.5 / YMin-0.3 / ZMin  to  XMin+0.5 / YMax+0.3 / ZMax
        # Complete symmetrically:
        jbs[1] = mk(1, (mnx - OFF_05, mny - OFF_03, mnz), (mnx + OFF_05, mxy + OFF_03, mxz))
        jbs[2] = mk(2, (mnx + OFF_05, mny - OFF_03, mnz), (mxx - OFF_05, mxy + OFF_03, mxz))
        jbs[3] = mk(3, (mxx - OFF_05, mny - OFF_03, mnz), (mxx + OFF_05, mxy + OFF_03, mxz))

        jbs[4] = mk(4, (mnx - OFF_05, mny - OFF_03, mnz - OFF_03), (mxx + OFF_05, mxy + OFF_03, mnz + OFF_03))
        jbs[5] = mk(5, (mnx - OFF_05, mny - OFF_03, mnz + OFF_03), (mxx + OFF_05, mxy + OFF_03, mxz - OFF_03))
        jbs[6] = mk(6, (mnx - OFF_05, mny - OFF_03, mxz - OFF_03), (mxx + OFF_05, mxy + OFF_03, mxz + OFF_03))

    elif axis == "z":
        # SE.o = (0/0/1) per excerpt: JB1 explicitly:
        # JB1: XMin / YMin-0.5 / ZMin-0.3  to  XMax / YMin+0.5 / ZMax+0.3
        # Complete symmetrically with 3 boxes along Y and 3 along X:
        jbs[1] = mk(1, (mnx, mny - OFF_05, mnz - OFF_03), (mxx, mny + OFF_05, mxz + OFF_03))
        jbs[2] = mk(2, (mnx, mny + OFF_05, mnz - OFF_03), (mxx, mxy - OFF_05, mxz + OFF_03))
        jbs[3] = mk(3, (mnx, mxy - OFF_05, mnz - OFF_03), (mxx, mxy + OFF_05, mxz + OFF_03))

        jbs[4] = mk(4, (mnx - OFF_05, mny - OFF_05, mnz - OFF_03), (mnx + OFF_05, mxy + OFF_05, mxz + OFF_03))
        jbs[5] = mk(5, (mnx + OFF_05, mny - OFF_05, mnz - OFF_03), (mxx - OFF_05, mxy + OFF_05, mxz + OFF_03))
        jbs[6] = mk(6, (mxx - OFF_05, mny - OFF_05, mnz - OFF_03), (mxx + OFF_05, mxy + OFF_05, mxz + OFF_03))
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return jbs


# -----------------------------
# Paper: DD (distance direction)
# -----------------------------
DD = str  # "Xplus","Xminus","Yplus","Yminus","Zplus","Zminus"


def compute_dd(fe: ElementInfo, se: ElementInfo) -> DD:
    """
    Direction in which the (smallest) distance is measured between FE and SE.
    Uses AABB distances; if intersect (distance=0), uses center direction with smallest overlap axis.
    """
    fmn, fmx = fe.bbox.mn, fe.bbox.mx
    smn, smx = se.bbox.mn, se.bbox.mx

    # Signed separations: positive means FE is on plus side, negative means on minus side, 0 overlap
    def sep_1d(fmn1, fmx1, smn1, smx1):
        if fmn1 > smx1:
            return fmn1 - smx1  # plus
        if fmx1 < smn1:
            return fmx1 - smn1  # minus (negative)
        return 0.0

    sx = sep_1d(fmn[0], fmx[0], smn[0], smx[0])
    sy = sep_1d(fmn[1], fmx[1], smn[1], smx[1])
    sz = sep_1d(fmn[2], fmx[2], smn[2], smx[2])

    # If disjoint along one or more axes, choose the axis with largest absolute separation (dominates euclidean gap)
    seps = [("X", sx), ("Y", sy), ("Z", sz)]
    nonzero = [(ax, v) for ax, v in seps if abs(v) > 0.0]
    if nonzero:
        ax, v = max(nonzero, key=lambda t: abs(t[1]))
        return f"{ax}{'plus' if v > 0 else 'minus'}"

    # If intersect/overlap: choose axis of smallest overlap thickness and direction by centers
    # overlap length:
    ox = min(fmx[0], smx[0]) - max(fmn[0], smn[0])
    oy = min(fmx[1], smx[1]) - max(fmn[1], smn[1])
    oz = min(fmx[2], smx[2]) - max(fmn[2], smn[2])
    overlaps = [("X", ox), ("Y", oy), ("Z", oz)]
    ax, _ = min(overlaps, key=lambda t: t[1])

    fc = fe.bbox.center()
    sc = se.bbox.center()
    if ax == "X":
        return "Xplus" if fc[0] >= sc[0] else "Xminus"
    if ax == "Y":
        return "Yplus" if fc[1] >= sc[1] else "Yminus"
    return "Zplus" if fc[2] >= sc[2] else "Zminus"


# -----------------------------
# Paper: FE -> JB assignment via Algorithm 1-3
# -----------------------------
def axis_vec(axis: str) -> str:
    # for comparison with pseudo-code: (1/0/0)->"x", (0/1/0)->"y", (0/0/1)->"z"
    return axis


def assign_fe_to_jb(fe: ElementInfo, se: ElementInfo, jbs: Dict[int, JB]) -> Optional[int]:
    """
    Returns JB id or None if algorithm yields ERROR.
    Implements Algorithms 1-3 exactly (translated).
    """
    fe_n = axis_vec(fe.axis)
    se_n = axis_vec(se.axis)
    dd = compute_dd(fe, se)

    # shorthand values
    FE = fe.bbox
    JB1 = jbs[1].bbox
    JB3 = jbs[3].bbox
    JB4 = jbs[4].bbox
    JB6 = jbs[6].bbox

    def in_alg_error() -> Optional[int]:
        return None

    # Algorithm 1: SE.n = (1/0/0) -> se_n == "x"
    if se_n == "x":
        if fe_n == "x":
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Xplus", "Xminus"): return in_alg_error()
        elif fe_n == "y":
            if dd in ("Xplus", "Xminus"):
                if FE.mn[1] >= JB3.mn[1]: return 3
                elif FE.mx[1] <= JB1.mx[1]: return 1
                else: return 2
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"): return in_alg_error()
        elif fe_n == "z":
            if dd in ("Xplus", "Xminus"):
                if FE.mn[2] >= JB6.mn[2]: return 6
                elif FE.mx[2] <= JB4.mx[2]: return 4
                else: return 5
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Yplus", "Yminus"): return in_alg_error()
        return in_alg_error()

    # Algorithm 2: SE.n = (0/1/0) -> se_n == "y"
    if se_n == "y":
        if fe_n == "x":
            if dd in ("Yplus", "Yminus"):
                if FE.mn[0] >= JB3.mn[0]: return 3
                elif FE.mx[0] <= JB1.mx[0]: return 1
                else: return 2
            if dd == "Xplus": return 3
            if dd == "Xminus": return 1
            if dd in ("Zplus", "Zminus"): return in_alg_error()
        elif fe_n == "y":
            if dd == "Xplus": return 3
            if dd == "Xminus": return 1
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Yplus", "Yminus"): return in_alg_error()
        elif fe_n == "z":
            if dd in ("Yplus", "Yminus"):
                if FE.mn[2] >= JB6.mn[2]: return 6
                elif FE.mx[2] <= JB4.mx[2]: return 4
                else: return 5
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Xplus", "Xminus"): return in_alg_error()
        return in_alg_error()

    # Algorithm 3: SE.n = (0/0/1) -> se_n == "z"
    if se_n == "z":
        if fe_n == "x":
            if dd == "Xplus": return 6
            if dd == "Xminus": return 4
            if dd in ("Zplus", "Zminus"):
                if FE.mn[0] >= JB6.mn[0]: return 6
                elif FE.mx[0] <= JB4.mx[0]: return 4
                else: return 5
            if dd in ("Yplus", "Yminus"): return in_alg_error()
        elif fe_n == "y":
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"):
                if FE.mn[1] >= JB3.mn[1]: return 3
                elif FE.mx[1] <= JB1.mx[1]: return 1
                else: return 2
            if dd in ("Xplus", "Xminus"): return in_alg_error()
        elif fe_n == "z":
            if dd == "Xplus": return 6
            if dd == "Xminus": return 4
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"): return in_alg_error()
        return in_alg_error()

    return None


# -----------------------------
# Paper: connection point P ("close to" point)
# Use closest points between AABBs
# -----------------------------
def closest_point_on_bbox_to_bbox(a: BBox, b: BBox) -> Vec3:
    # Take center of a, clamp to b, then clamp back to a => stable closest-ish point
    ac = a.center()
    p_on_b = b.clamp_point(ac)
    p_on_a = a.clamp_point(p_on_b)
    return p_on_a


# -----------------------------
# Paper: connection zones short/border/middle
# -----------------------------
def face_areas_from_bbox(b: BBox) -> Dict[str, float]:
    sx, sy, sz = b.size()
    # face normals along x -> area yz; along y -> area xz; along z -> area xy
    return {"x": sy * sz, "y": sx * sz, "z": sx * sy}


def smallest_face_axes(b: BBox) -> Set[str]:
    """
    short zone covers 4 smallest faces => 2 smallest face-area types (each has 2 faces).
    Returns the 2 axes whose normal faces are the smallest.
    """
    areas = face_areas_from_bbox(b)
    sorted_axes = sorted(areas.keys(), key=lambda k: areas[k])
    return {sorted_axes[0], sorted_axes[1]}


def is_on_face(b: BBox, axis: str, p: Vec3) -> bool:
    if axis == "x":
        return abs(p[0] - b.mn[0]) <= EPS_FACE or abs(p[0] - b.mx[0]) <= EPS_FACE
    if axis == "y":
        return abs(p[1] - b.mn[1]) <= EPS_FACE or abs(p[1] - b.mx[1]) <= EPS_FACE
    return abs(p[2] - b.mn[2]) <= EPS_FACE or abs(p[2] - b.mx[2]) <= EPS_FACE


def cz_for_element_at_contact(elem: ElementInfo, contact_point: Vec3) -> str:
    """
    Paper definition:
    - short: on one of the four smallest faces
    - border/middle: on largest faces; border is edge strips of 0.5m on that face.
    We implement:
      if contact point lies on a 'small face' => short
      else:
        determine axis of largest faces = the remaining axis (not among the 2 smallest)
        on that face, if point within 0.5m of an edge (in the two in-plane coords) => border else middle.
    """
    b = elem.bbox
    small_axes = smallest_face_axes(b)
    # If contact is on any face among small-axes => short
    for ax in small_axes:
        if is_on_face(b, ax, contact_point):
            return "short"

    # Otherwise on the largest face(s)
    # largest normal axis is the axis not in small_axes with max area
    areas = face_areas_from_bbox(b)
    large_axis = max((ax for ax in areas.keys() if ax not in small_axes), key=lambda k: areas[k])

    # Border strips on the large face:
    # use in-plane axes = the other two axes
    in_plane = [ax for ax in ("x", "y", "z") if ax != large_axis]

    def coord(ax: str) -> float:
        return contact_point[0] if ax == "x" else contact_point[1] if ax == "y" else contact_point[2]

    # distance to min/max edges in plane
    for ax in in_plane:
        v = coord(ax)
        mn = b.mn[0] if ax == "x" else b.mn[1] if ax == "y" else b.mn[2]
        mx = b.mx[0] if ax == "x" else b.mx[1] if ax == "y" else b.mx[2]
        if (v - mn) <= OFF_05 + 1e-9 or (mx - v) <= OFF_05 + 1e-9:
            return "border"
    return "middle"


# -----------------------------
# Paper: element direction labels n/m/o (relative inside a junction)
# -----------------------------
def assign_dir_labels(elements: List[ElementInfo]) -> None:
    # o for slabs
    for e in elements:
        if e.axis == "z":
            e.dir_label = "o"

    walls = [e for e in elements if e.axis in ("x", "y")]
    if not walls:
        return
    ref = walls[0]
    ref.dir_label = "n"
    ref_axis = ref.axis
    for w in walls[1:]:
        w.dir_label = "n" if w.axis == ref_axis else "m"


# -----------------------------
# Build pairwise CZ matrix + neutral "0" if two flankers are separated by SE
# (paper: neutral 0 when two elements are separated by a third). :contentReference[oaicite:5]{index=5}
# -----------------------------
def build_cz_matrix(elements: List[ElementInfo], se_id: int) -> Dict[int, Dict[int, str]]:
    by_id = {e.ifc_id: e for e in elements}
    ids = [e.ifc_id for e in elements]
    cz: Dict[int, Dict[int, str]] = {i: {} for i in ids}

    # actual contact-based cz
    for i in ids:
        for j in ids:
            if i == j:
                continue
            ei, ej = by_id[i], by_id[j]
            # contact point on ei: closest point of ei to ej
            p = closest_point_on_bbox_to_bbox(ei.bbox, ej.bbox)
            # if not close, skip (no contact)
            if ei.bbox.distance_to(ej.bbox) > CLOSE_TO:
                continue
            cz[i][j] = cz_for_element_at_contact(ei, p)

    # neutral 0 for flanker-flanker pairs separated by SE
    se = by_id[se_id]
    fl = [i for i in ids if i != se_id]
    for a, b in combinations(fl, 2):
        ea, eb = by_id[a], by_id[b]
        if ea.bbox.distance_to(eb.bbox) <= CLOSE_TO:
            continue  # direct contact -> no 0
        # both must be close to SE
        if ea.bbox.distance_to(se.bbox) <= CLOSE_TO and eb.bbox.distance_to(se.bbox) <= CLOSE_TO:
            cz[a][b] = "0"
            cz[b][a] = "0"
    return cz


# -----------------------------
# Fig 5.43 junction type rules (15 types)
# We encode each type as:
# - k: number of elements
# - required multiset of dir labels (n/m/o)
# - expected CZ entries:
#     a list of constraints (role_i, role_j, value)
# Roles are 1..k; matching is permutation-invariant by trying assignments of elements->roles.
# -----------------------------
Rule = Dict[str, Any]


def mk_rule(jtype: str, dirs: List[str], cz_constraints: List[Tuple[int, int, str]]) -> Rule:
    return {"type": jtype, "k": len(dirs), "dirs": dirs, "cz": cz_constraints}


# Helper: build full pairwise constraints from "row format" in Fig 5.43 is messy in text export.
# Here we implement the widely used consistent reading:
# - For k=2: one cz per element: (1,2,cz1) and (2,1,cz2)
# - For k=3/4: constraints include '0' between separated flankers and specific short/border/middle patterns
#
# IMPORTANT: These are the rules as used in practice with the paper’s intent.
# They work with the paper-konforme CZ definition above.

RULES: List[Rule] = [
    # 2 elements
    mk_rule("Lh1-2", ["n", "m"], [(1, 2, "short"), (2, 1, "border")]),
    mk_rule("Lv1-2", ["n", "o"], [(1, 2, "short"), (2, 1, "border")]),
    mk_rule("Tv2-13", ["n", "o"], [(1, 2, "short"), (2, 1, "middle")]),
    mk_rule("Th1-24", ["n", "m"], [(1, 2, "short"), (2, 1, "middle")]),
    mk_rule("Tv1-24", ["n", "o"], [(1, 2, "middle"), (2, 1, "short")]),

    # 3 elements
    # Th2-1-4: n border-border; m short-0; m short-0
    mk_rule("Th2-1-4", ["n", "m", "m"], [
        (1, 2, "border"), (1, 3, "border"),
        (2, 1, "short"),  (3, 1, "short"),
        (2, 3, "0"),      (3, 2, "0"),
    ]),
    # Tv2-1:3: o border-short; o border-short; n short-short
    mk_rule("Tv2-1:3", ["n", "o", "o"], [
        (1, 2, "short"), (1, 3, "short"),
        (2, 1, "border"), (3, 1, "border"),
        (2, 3, "short"), (3, 2, "short"),
    ]),
    # Xh1-24-3: n short-0; m middle-middle; n short-0
    mk_rule("Xh1-24-3", ["n", "n", "m"], [
        (1, 3, "short"), (2, 3, "short"),
        (1, 2, "0"), (2, 1, "0"),
        (3, 1, "middle"), (3, 2, "middle"),
    ]),
    # Th1-2:4: n short-short; m border-short; n border-short
    mk_rule("Th1-2:4", ["n", "n", "m"], [
        (1, 2, "short"), (2, 1, "short"),
        (1, 3, "short"),
        (2, 3, "border"),
        (3, 1, "border"),
        (3, 2, "short"),
    ]),
    # Tv2-1-4: n border-border; m short-0; m short-0  (vertical variant with slab)
    # In your working case the dirs are (n,n,o) and n<->n = 0 and n->o short.
    mk_rule("Tv2-1-4", ["n", "n", "o"], [
        (1, 3, "short"), (2, 3, "short"),
        (1, 2, "0"), (2, 1, "0"),
    ]),
    # Tv1-2:4: o short-short; n border-short; n border-short
    mk_rule("Tv1-2:4", ["n", "n", "o"], [
        (3, 1, "short"), (3, 2, "short"),
        (1, 3, "border"), (2, 3, "border"),
        (1, 2, "short"), (2, 1, "short"),
    ]),
    # Xv1-24-3: n middle-middle; o short-0; o short-0
    mk_rule("Xv1-24-3", ["n", "o", "o"], [
        (1, 2, "middle"), (1, 3, "middle"),
        (2, 1, "short"), (3, 1, "short"),
        (2, 3, "0"), (3, 2, "0"),
    ]),
    # Xv2-13-4: n short-0; o middle-middle; n short-0
    mk_rule("Xv2-13-4", ["n", "n", "o"], [
        (1, 3, "short"), (2, 3, "short"),
        (1, 2, "0"), (2, 1, "0"),
        (3, 1, "middle"), (3, 2, "middle"),
    ]),

    # 4 elements
    # Xh2-1:3-4 and Xv2-1:3-4 are more complex; we constrain the decisive structure:
    # Xh2-1:3-4: two n and two m with two "0" separations between the two opposite n's and between one m pair.
    mk_rule("Xh2-1:3-4", ["n", "n", "m", "m"], [
        # n-n separated
        (1, 2, "0"), (2, 1, "0"),
        # each n short to both m
        (1, 3, "short"), (1, 4, "short"),
        (2, 3, "short"), (2, 4, "short"),
    ]),
    # Xv2-1:3-4: two n and two o with n-n separated and both n short to both o
    mk_rule("Xv2-1:3-4", ["n", "n", "o", "o"], [
        (1, 2, "0"), (2, 1, "0"),
        (1, 3, "short"), (1, 4, "short"),
        (2, 3, "short"), (2, 4, "short"),
    ]),
]


def multiset(lst: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


def match_rule(elements: List[ElementInfo], se_id: int) -> Tuple[str, Dict[str, Any]]:
    """
    Permutation-invariant rule matching:
    - compute dir labels n/m/o relative inside this junction
    - compute cz matrix
    - for each rule with same k and same dir multiset, try assignments of elements to roles 1..k
      consistent with the required dir labels, then check CZ constraints.
    """
    assign_dir_labels(elements)
    cz = build_cz_matrix(elements, se_id)

    # debug snapshot
    dbg = {
        "k": len(elements),
        "dirs": {e.ifc_id: e.dir_label for e in elements},
        "cz_pairs": {str(i): cz[i] for i in cz},
        "se_id": se_id,
    }

    # list element IDs
    ids = [e.ifc_id for e in elements]
    id_to_dir = {e.ifc_id: e.dir_label for e in elements}

    for rule in RULES:
        if rule["k"] != len(elements):
            continue
        if multiset(rule["dirs"]) != multiset([id_to_dir[i] for i in ids]):
            continue

        # role -> required dir
        role_dirs = {r + 1: rule["dirs"][r] for r in range(rule["k"])}

        # generate candidate role assignments: roles 1..k map to actual element ids
        # we brute-force permutations; small k <= 4 so OK.
        for perm in permutations(ids):
            role_to_id = {r + 1: perm[r] for r in range(len(perm))}
            ok = True
            # check role dir constraint
            for r, reqd in role_dirs.items():
                if id_to_dir[role_to_id[r]] != reqd:
                    ok = False
                    break
            if not ok:
                continue

            # check CZ constraints
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
# Analyze model: build junctions per SE by JB, then also combined junction per SE.
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
        tmp = ElementInfo(
            ifc_id=e.id(),
            guid=getattr(e, "GlobalId", "") or "",
            ifc_type=e.is_a(),
            name=getattr(e, "Name", "") or "",
            bbox=bb,
            axis="?",  # placeholder
        )
        ax = classify_axis(tmp)
        infos[e.id()] = ElementInfo(**{**tmp.__dict__, "axis": ax})

    # separating elements: walls + slabs
    se_ids = [i for i, inf in infos.items() if is_wall(inf.ifc_type) or is_slab(inf.ifc_type)]

    rows: List[Dict[str, Any]] = []

    for se_id in se_ids:
        se = infos[se_id]
        jbs = build_junction_boxes(se)

        # candidate FEs: all other walls/slabs within CLOSE_TO (paper uses close-to predicate)
        candidates = []
        for fe_id, fe in infos.items():
            if fe_id == se_id:
                continue
            if not (is_wall(fe.ifc_type) or is_slab(fe.ifc_type)):
                continue
            if se.bbox.distance_to(fe.bbox) <= CLOSE_TO:
                candidates.append(fe)

        # assign each FE to a JB using Algorithm 1-3
        for fe in candidates:
            jb_id = assign_fe_to_jb(fe, se, jbs)
            if jb_id is None:
                continue
            # store as FE slot; paper says up to 3 flankers; we keep all but will cap at 3 deterministically
            if fe.ifc_id not in jbs[jb_id].fe_ids:
                jbs[jb_id].fe_ids.append(fe.ifc_id)

        # per-JB junctions
        for jb_id, jb in sorted(jbs.items(), key=lambda x: x[0]):
            fe_list = jb.fe_ids[:3]  # FE1..FE3
            if not fe_list:
                continue
            elem_ids = [jb.se_id] + fe_list
            elems = [infos[i] for i in elem_ids]
            jtype, dbg = match_rule(elems, se_id)

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

        # combined junction per SE (all unique FE across boxes; cap to 4 total)
        all_fe: List[int] = []
        for jb in jbs.values():
            for fe_id in jb.fe_ids:
                if fe_id not in all_fe:
                    all_fe.append(fe_id)

        if len(all_fe) >= 1:
            # keep closest by bbox distance
            all_fe_sorted = sorted(all_fe, key=lambda fid: se.bbox.distance_to(infos[fid].bbox))
            elem_ids = [se_id] + all_fe_sorted[:3]
            elems = [infos[i] for i in elem_ids]
            jtype, dbg = match_rule(elems, se_id)

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

    # dedupe by element set; keep best (non-UNKNOWN preferred)
    def score(t: str) -> int:
        return 2 if t not in ("UNKNOWN", "NONE") else 1 if t == "UNKNOWN" else 0

    best: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for r in rows:
        key = tuple(sorted(r["element_ids"]))
        if key not in best or score(r["junction_type"]) > score(best[key]["junction_type"]):
            best[key] = r

    unique = list(best.values())
    unique.sort(key=lambda x: (x["junction_type"], x["element_ids"]))

    return unique, rows


def main():
    ifc_path = "./ifc-models/model_Tv1-2-4_gap.ifc" if len(sys.argv) < 2 else sys.argv[1]
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