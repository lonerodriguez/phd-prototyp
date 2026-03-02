#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-konforme Stoßstellenanalyse (Hellwig Diss-Auszug) – OPTIMIERT

Wichtiger Fix (paper-konform):
- Connection Zone (CZ) für das Trennelement (SE) wird NICHT mehr über "kleinste Flächen"
  bestimmt, sondern wie in Abb. 5.39–5.41: über Junction Box (JB) + Randstreifen 0.5m.
  => Tv1-24 klappt: SE(n)->FE(o)=middle und FE(o)->SE(n)=short.

Features:
- DD robust (nächstgelegene SE-Fläche)
- Kontaktfilter gegen Kante/Punkt (Mindestüberlappung) + Mini-Gap-Toleranz
- Wall–Wall stacked contact erkannt
- FE->JB Zuordnung per Algorithmus 1–3
- CZ:
  * SE-seitig paper-konform (JB + 0.5m)
  * FE-seitig BBox-Flächenlogik (short/border/middle)
- Junction type matching permutation-invariant
- Ausgabe: nur maximal (größte) Junctions (Superset wins)

Outputs:
  junctions_unique.json
  junctions_debug.json
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
OFF_03 = 0.30        # +/-0.3 m
OFF_05 = 0.50        # +/-0.5 m  (border strip)
CLOSE_TO = 0.30      # "close to" threshold (meters)

# Reject edge-only / point contacts:
MIN_OVERLAP_LEN = 0.05   # 5 cm
MIN_OVERLAP_AREA = 0.01  # 100 cm²

EPS_FACE = 1e-6
EDGE_GAP_TOL = 0.02      # 2 cm: erlaubt "Berührung/Minigap" für line-contact

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
    axis: str               # 'x','y','z' paper abstraction
    dir_label: str = ""     # 'n','m','o' inside a junction


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


def classify_axis_from_bbox(ifc_type: str, b: BBox) -> str:
    sx, sy, _ = b.size()
    if is_slab(ifc_type):
        return "z"
    return "x" if sx >= sy else "y"


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
# Junction Boxes (JB1..JB6)
# -----------------------------
def build_junction_boxes(se: ElementInfo) -> Dict[int, JB]:
    mnx, mny, mnz = se.bbox.mn
    mxx, mxy, mxz = se.bbox.mx

    def mk(jb_id: int, mn: Vec3, mx: Vec3) -> JB:
        return JB(jb_id=jb_id, bbox=BBox(mn, mx), se_id=se.ifc_id, fe_ids=[])

    axis = se.axis
    jbs: Dict[int, JB] = {}

    if axis == "x":
        jbs[1] = mk(1, (mnx - OFF_03, mny - OFF_05, mnz), (mxx + OFF_03, mny + OFF_05, mxz))
        jbs[2] = mk(2, (mnx - OFF_03, mny + OFF_05, mnz), (mxx + OFF_03, mxy - OFF_05, mxz))
        jbs[3] = mk(3, (mnx - OFF_03, mxy - OFF_05, mnz), (mxx + OFF_03, mxy + OFF_05, mxz))

        jbs[4] = mk(4, (mnx - OFF_03, mny - OFF_05, mnz - OFF_03), (mxx + OFF_03, mxy + OFF_05, mnz + OFF_03))
        jbs[5] = mk(5, (mnx - OFF_03, mny - OFF_05, mnz + OFF_03), (mxx + OFF_03, mxy + OFF_05, mxz - OFF_03))
        jbs[6] = mk(6, (mnx - OFF_03, mny - OFF_05, mxz - OFF_03), (mxx + OFF_03, mxy + OFF_05, mxz + OFF_03))

    elif axis == "y":
        jbs[1] = mk(1, (mnx - OFF_05, mny - OFF_03, mnz), (mnx + OFF_05, mxy + OFF_03, mxz))
        jbs[2] = mk(2, (mnx + OFF_05, mny - OFF_03, mnz), (mxx - OFF_05, mxy + OFF_03, mxz))
        jbs[3] = mk(3, (mxx - OFF_05, mny - OFF_03, mnz), (mxx + OFF_05, mxy + OFF_03, mxz))

        jbs[4] = mk(4, (mnx - OFF_05, mny - OFF_03, mnz - OFF_03), (mxx + OFF_05, mxy + OFF_03, mnz + OFF_03))
        jbs[5] = mk(5, (mnx - OFF_05, mny - OFF_03, mnz + OFF_03), (mxx + OFF_05, mxy + OFF_03, mxz - OFF_03))
        jbs[6] = mk(6, (mnx - OFF_05, mny - OFF_03, mxz - OFF_03), (mxx + OFF_05, mxy + OFF_03, mxz + OFF_03))

    elif axis == "z":
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
# DD
# -----------------------------
def compute_dd(fe: ElementInfo, se: ElementInfo) -> DD:
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
    dd, _ = min(face_dists, key=lambda t: t[1])
    return dd


# -----------------------------
# FE -> JB assignment (Algorithmus 1–3)
# -----------------------------
def assign_fe_to_jb(fe: ElementInfo, se: ElementInfo, jbs: Dict[int, JB]) -> Optional[int]:
    fe_n = fe.axis
    se_n = se.axis
    dd = compute_dd(fe, se)

    FE = fe.bbox
    JB1 = jbs[1].bbox
    JB3 = jbs[3].bbox
    JB4 = jbs[4].bbox
    JB6 = jbs[6].bbox

    def err():
        return None

    # Algorithm 1: SE.n = x
    if se_n == "x":
        if fe_n == "x":
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            return err()
        elif fe_n == "y":
            if dd in ("Xplus", "Xminus"):
                if FE.mn[1] >= JB3.mn[1]: return 3
                if FE.mx[1] <= JB1.mx[1]: return 1
                return 2
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            return err()
        elif fe_n == "z":
            if dd in ("Xplus", "Xminus"):
                if FE.mn[2] >= JB6.mn[2]: return 6
                if FE.mx[2] <= JB4.mx[2]: return 4
                return 5
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            return err()
        return err()

    # Algorithm 2: SE.n = y
    if se_n == "y":
        if fe_n == "x":
            if dd in ("Yplus", "Yminus"):
                if FE.mn[0] >= JB3.mn[0]: return 3
                if FE.mx[0] <= JB1.mx[0]: return 1
                return 2
            if dd == "Xplus": return 3
            if dd == "Xminus": return 1
            return err()
        elif fe_n == "y":
            if dd == "Xplus": return 3
            if dd == "Xminus": return 1
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            return err()
        elif fe_n == "z":
            if dd in ("Yplus", "Yminus"):
                if FE.mn[2] >= JB6.mn[2]: return 6
                if FE.mx[2] <= JB4.mx[2]: return 4
                return 5
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            return err()
        return err()

    # Algorithm 3: SE.n = z
    if se_n == "z":
        if fe_n == "x":
            if dd == "Xplus": return 6
            if dd == "Xminus": return 4
            if dd in ("Zplus", "Zminus"):
                if FE.mn[0] >= JB6.mn[0]: return 6
                if FE.mx[0] <= JB4.mx[0]: return 4
                return 5
            return err()
        elif fe_n == "y":
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"):
                if FE.mn[1] >= JB3.mn[1]: return 3
                if FE.mx[1] <= JB1.mx[1]: return 1
                return 2
            return err()
        elif fe_n == "z":
            if dd == "Xplus": return 6
            if dd == "Xminus": return 4
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            return err()
        return err()

    return None


# -----------------------------
# Connection zones
# -----------------------------
def face_areas_from_bbox(b: BBox) -> Dict[str, float]:
    sx, sy, sz = b.size()
    return {"x": sy * sz, "y": sx * sz, "z": sx * sy}


def smallest_face_axes(b: BBox) -> Set[str]:
    areas = face_areas_from_bbox(b)
    sorted_axes = sorted(areas.keys(), key=lambda k: areas[k])
    return {sorted_axes[0], sorted_axes[1]}


def is_on_face(b: BBox, axis: str, p: Vec3) -> bool:
    if axis == "x":
        return abs(p[0] - b.mn[0]) <= EPS_FACE or abs(p[0] - b.mx[0]) <= EPS_FACE
    if axis == "y":
        return abs(p[1] - b.mn[1]) <= EPS_FACE or abs(p[1] - b.mx[1]) <= EPS_FACE
    return abs(p[2] - b.mn[2]) <= EPS_FACE or abs(p[2] - b.mx[2]) <= EPS_FACE


def cz_for_element_at_contact(elem: ElementInfo, p_on_elem: Vec3) -> str:
    """
    FE-seitig: BBox-Flächenlogik (wie bisher).
    """
    b = elem.bbox
    small_axes = smallest_face_axes(b)

    for ax in small_axes:
        if is_on_face(b, ax, p_on_elem):
            return "short"

    areas = face_areas_from_bbox(b)
    large_axis = max((ax for ax in areas.keys() if ax not in small_axes), key=lambda k: areas[k])
    in_plane = [ax for ax in ("x", "y", "z") if ax != large_axis]

    def coord(ax: str) -> float:
        return p_on_elem[0] if ax == "x" else p_on_elem[1] if ax == "y" else p_on_elem[2]

    for ax in in_plane:
        v = coord(ax)
        mn = b.mn[0] if ax == "x" else b.mn[1] if ax == "y" else b.mn[2]
        mx = b.mx[0] if ax == "x" else b.mx[1] if ax == "y" else b.mx[2]
        if (v - mn) <= OFF_05 + 1e-9 or (mx - v) <= OFF_05 + 1e-9:
            return "border"
    return "middle"


def cz_on_se_paper(se: ElementInfo, fe: ElementInfo, fe_jb: Optional[int]) -> str:
    """
    SE-seitig: paper-konform nach Abb. 5.39–5.41:
    border, wenn Kontaktpunkt P im Randstreifen (0.5m) liegt – abhängig von
    SE-Achse und FE-Achse und JB-Lage des FE.
    Sonst middle.

    WICHTIG: In diesen orthogonalen Fällen liefert das Paper für SE NICHT 'short',
    sondern border/middle.
    """
    if fe_jb is None:
        return "middle"

    # P: "beliebiger Punkt des flankierenden Bauteils", close to
    # Praktisch: projiziere FE-center auf SE (auf SE-BBox geklemmt)
    p = se.bbox.clamp_point(fe.bbox.center())
    mnx, mny, mnz = se.bbox.mn
    mxx, mxy, mxz = se.bbox.mx

    # SE = wall in x (n=(1/0/0))  Abb. 5.39
    if se.axis == "x":
        # FE = wall in y (m)
        if fe.axis == "y":
            if fe_jb == 1:  # JB1 -> Y.Min..Y.Min+0.5
                return "border" if (mny - 1e-9) <= p[1] <= (mny + OFF_05 + 1e-9) else "middle"
            if fe_jb == 3:  # JB3 -> Y.Max-0.5..Y.Max
                return "border" if (mxy - OFF_05 - 1e-9) <= p[1] <= (mxy + 1e-9) else "middle"
            return "middle"
        # FE = slab (o)
        if fe.axis == "z":
            if fe_jb == 4:  # JB4 -> Z.Min..Z.Min+0.5
                return "border" if (mnz - 1e-9) <= p[2] <= (mnz + OFF_05 + 1e-9) else "middle"
            if fe_jb == 6:  # JB6 -> Z.Max-0.5..Z.Max
                return "border" if (mxz - OFF_05 - 1e-9) <= p[2] <= (mxz + 1e-9) else "middle"
            return "middle"

    # SE = wall in y (m=(0/1/0))  Abb. 5.40
    if se.axis == "y":
        # FE = wall in x (n)
        if fe.axis == "x":
            if fe_jb == 1:  # JB1 -> X.Min..X.Min+0.5
                return "border" if (mnx - 1e-9) <= p[0] <= (mnx + OFF_05 + 1e-9) else "middle"
            if fe_jb == 3:  # JB3 -> X.Max-0.5..X.Max
                return "border" if (mxx - OFF_05 - 1e-9) <= p[0] <= (mxx + 1e-9) else "middle"
            return "middle"
        # FE = slab (o)
        if fe.axis == "z":
            if fe_jb == 4:
                return "border" if (mnz - 1e-9) <= p[2] <= (mnz + OFF_05 + 1e-9) else "middle"
            if fe_jb == 6:
                return "border" if (mxz - OFF_05 - 1e-9) <= p[2] <= (mxz + 1e-9) else "middle"
            return "middle"

    # SE = slab (o=(0/0/1))  Abb. 5.41
    if se.axis == "z":
        # FE = wall in y (m)
        if fe.axis == "y":
            if fe_jb == 1:
                return "border" if (mny - 1e-9) <= p[1] <= (mny + OFF_05 + 1e-9) else "middle"
            if fe_jb == 3:
                return "border" if (mxy - OFF_05 - 1e-9) <= p[1] <= (mxy + 1e-9) else "middle"
            return "middle"
        # FE = wall in x (n) -> in JB4/JB6 über X
        if fe.axis == "x":
            if fe_jb == 4:
                return "border" if (mnx - 1e-9) <= p[0] <= (mnx + OFF_05 + 1e-9) else "middle"
            if fe_jb == 6:
                return "border" if (mxx - OFF_05 - 1e-9) <= p[0] <= (mxx + 1e-9) else "middle"
            return "middle"

    # Default (parallel/unklar)
    return "middle"


def contact_point_on_element_bbox(ei: ElementInfo, ej: ElementInfo) -> Vec3:
    """
    FE-seitiger Kontaktpunkt (für cz_for_element_at_contact):
    projiziere ej.center auf ei bbox.
    """
    return ei.bbox.clamp_point(ej.bbox.center())


def assign_dir_labels(elements: List[ElementInfo]) -> None:
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


def build_cz_matrix(elements: List[ElementInfo], se_id: int, fe_to_jb: Dict[int, int]) -> Dict[int, Dict[int, str]]:
    """
    CZ-Matrix:
    - Für i == se_id: paper-konform (JB + 0.5m border strip)
    - Für i != se_id: BBox-Flächenlogik (short/border/middle)
    """
    by_id = {e.ifc_id: e for e in elements}
    ids = [e.ifc_id for e in elements]
    cz: Dict[int, Dict[int, str]] = {i: {} for i in ids}

    se = by_id[se_id]

    for i in ids:
        for j in ids:
            if i == j:
                continue
            ei, ej = by_id[i], by_id[j]
            if not has_sufficient_contact(ej, ei):
                continue

            if i == se_id:
                jb = fe_to_jb.get(j)
                cz[i][j] = cz_on_se_paper(se, ej, jb)
            else:
                p = contact_point_on_element_bbox(ei, ej)
                cz[i][j] = cz_for_element_at_contact(ei, p)

    # neutral 0 for flanker-flanker separated by SE (paper cases with ":")
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

    mk_rule("Tv2-1-4", ["n", "n", "o"], [
        (1, 3, "short"), (2, 3, "short"),
        (1, 2, "0"), (2, 1, "0"),
    ]),

    mk_rule("Tv1-2-4", ["n", "n", "o"], [
        (1, 3, "short"), (2, 3, "short"),
        (3, 1, "border"), (3, 2, "border"),
        (1, 2, "short"), (2, 1, "short"),
    ]),
    mk_rule("Tv1-2:4", ["n", "n", "o"], [
        (1, 3, "short"), (2, 3, "short"),
        (3, 1, "border"), (3, 2, "border"),
        (1, 2, "0"), (2, 1, "0"),
    ]),
]


def multiset(lst: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


def match_rule(elements: List[ElementInfo], se_id: int, fe_to_jb: Dict[int, int]) -> Tuple[str, Dict[str, Any]]:
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
# Post processing: keep only maximal junctions (superset wins)
# -----------------------------
def junction_score(j: Dict[str, Any]) -> Tuple[int, int]:
    jt = j.get("junction_type", "UNKNOWN")
    recognized = 1 if jt not in ("UNKNOWN", "NONE") else 0
    size = len(j.get("element_ids", []))
    return (recognized, size)


def keep_only_maximal_junctions(junctions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sets = [(frozenset(j["element_ids"]), j) for j in junctions]
    sets.sort(key=lambda x: (junction_score(x[1])[0], junction_score(x[1])[1]), reverse=True)

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
        ax = classify_axis_from_bbox(e.is_a(), bb)
        infos[e.id()] = ElementInfo(
            ifc_id=e.id(),
            guid=getattr(e, "GlobalId", "") or "",
            ifc_type=e.is_a(),
            name=getattr(e, "Name", "") or "",
            bbox=bb,
            axis=ax,
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

        # JB assignment + mapping
        fe_to_jb_all: Dict[int, int] = {}
        for fe in candidates:
            jb_id = assign_fe_to_jb(fe, se, jbs)
            if jb_id is None:
                continue
            fe_to_jb_all[fe.ifc_id] = jb_id
            if fe.ifc_id not in jbs[jb_id].fe_ids:
                jbs[jb_id].fe_ids.append(fe.ifc_id)

        # per-JB junctions
        for jb_id, jb in sorted(jbs.items(), key=lambda x: x[0]):
            fe_list = jb.fe_ids[:3]
            if not fe_list:
                continue
            elem_ids = [jb.se_id] + fe_list
            elems = [infos[i] for i in elem_ids]

            fe_to_jb = {fid: fe_to_jb_all.get(fid, jb_id) for fid in fe_list}
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

        # COMBINED: closest 3
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

    def base_score(t: str) -> int:
        return 2 if t not in ("UNKNOWN", "NONE") else 1 if t == "UNKNOWN" else 0

    best: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for r in rows:
        key = tuple(sorted(r["element_ids"]))
        if key not in best or base_score(r["junction_type"]) > base_score(best[key]["junction_type"]):
            best[key] = r

    unique = list(best.values())
    unique = keep_only_maximal_junctions(unique)
    unique.sort(key=lambda x: (x["junction_type"], x["element_ids"]))
    return unique, rows


def main():
    ifc_path = "./ifc-models/Tv1-24.ifc" if len(sys.argv) < 2 else sys.argv[1]
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