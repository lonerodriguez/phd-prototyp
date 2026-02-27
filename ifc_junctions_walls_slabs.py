#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-konforme Stoßstellenanalyse (Hellwig Diss-Auszug) – OPTIMIERT

Änderungen (aktueller Stand):
- DD (distance direction) robust über nächstgelegene SE-Fläche am Kontaktpunkt
- Kandidaten-/CZ-Filter gegen "nur Kante / nur Punkt"-Kontakte (Mindest-Überlappung)
- FE->JB Zuordnung per Algorithmus 1–3
- Connection Zones short/border/middle nach BBox-Flächen-Logik + border strip 0.5m
- Junction type matching permutation-invariant
- Ausgabe: nur maximal (größte) Junctions (Superset wins)

Usage:
  python ifc_junctions_paper_optimized.py            # reads ./model.ifc
  python ifc_junctions_paper_optimized.py my.ifc

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
OFF_05 = 0.50        # +/-0.5 m
CLOSE_TO = 0.30      # "close to" threshold

# Reject edge-only / point contacts:
MIN_OVERLAP_LEN = 0.05   # 5 cm
MIN_OVERLAP_AREA = 0.01  # 100 cm²

EPS_FACE = 1e-6

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


def overlap_lengths(a: BBox, b: BBox) -> Tuple[float, float, float]:
    ox = overlap_1d(a.mn[0], a.mx[0], b.mn[0], b.mx[0])
    oy = overlap_1d(a.mn[1], a.mx[1], b.mn[1], b.mx[1])
    oz = overlap_1d(a.mn[2], a.mx[2], b.mn[2], b.mx[2])
    return ox, oy, oz


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
    dir_label: str = ""     # 'n','m','o' relative inside a junction


@dataclass
class JB:
    jb_id: int
    bbox: BBox
    se_id: int
    fe_ids: List[int]


# -----------------------------
# IFC helpers
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
    """Paper assumption: axis-aligned elements; walls along x or y, slabs along z."""
    sx, sy, _ = b.size()
    if is_slab(ifc_type):
        return "z"
    return "x" if sx >= sy else "y"


# -----------------------------
# Candidate/contact filter (implements your point 6)
# -----------------------------
def has_sufficient_contact(fe: ElementInfo, se: ElementInfo) -> bool:
    """
    Reject edge-only/point contacts.
    Accept only if there is sufficient overlap in the expected contact plane.
    """
    if se.bbox.distance_to(fe.bbox) > CLOSE_TO:
        return False

    ox, oy, oz = overlap_lengths(se.bbox, fe.bbox)

    se_is_slab = is_slab(se.ifc_type)
    fe_is_slab = is_slab(fe.ifc_type)
    se_is_wall = is_wall(se.ifc_type)
    fe_is_wall = is_wall(fe.ifc_type)

    # Wall–Slab: contact plane should be XY (overlap in X and Y)
    if (se_is_wall and fe_is_slab) or (se_is_slab and fe_is_wall):
        area_xy = ox * oy
        return (ox >= MIN_OVERLAP_LEN and oy >= MIN_OVERLAP_LEN and area_xy >= MIN_OVERLAP_AREA)

    # Wall–Wall: require vertical overlap + one horizontal overlap
    if se_is_wall and fe_is_wall:
        if oz < MIN_OVERLAP_LEN:
            return False
        return (ox >= MIN_OVERLAP_LEN) or (oy >= MIN_OVERLAP_LEN)

    # Slab–Slab: require XY overlap
    if se_is_slab and fe_is_slab:
        area_xy = ox * oy
        return (ox >= MIN_OVERLAP_LEN and oy >= MIN_OVERLAP_LEN and area_xy >= MIN_OVERLAP_AREA)

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
# DD (robust contact-face based)
# -----------------------------
def closest_point_on_se_from_fe(se: BBox, fe: BBox) -> Vec3:
    return se.clamp_point(fe.center())


def compute_dd(fe: ElementInfo, se: ElementInfo) -> DD:
    p = closest_point_on_se_from_fe(se.bbox, fe.bbox)
    smn, smx = se.bbox.mn, se.bbox.mx

    d_xmin = abs(p[0] - smn[0])
    d_xmax = abs(smx[0] - p[0])
    d_ymin = abs(p[1] - smn[1])
    d_ymax = abs(smx[1] - p[1])
    d_zmin = abs(p[2] - smn[2])
    d_zmax = abs(smx[2] - p[2])

    face_dists = [
        ("Xminus", d_xmin),
        ("Xplus", d_xmax),
        ("Yminus", d_ymin),
        ("Yplus", d_ymax),
        ("Zminus", d_zmin),
        ("Zplus", d_zmax),
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
            if dd in ("Xplus", "Xminus"): return err()
        elif fe_n == "y":
            if dd in ("Xplus", "Xminus"):
                if FE.mn[1] >= JB3.mn[1]: return 3
                if FE.mx[1] <= JB1.mx[1]: return 1
                return 2
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"): return err()
        elif fe_n == "z":
            if dd in ("Xplus", "Xminus"):
                if FE.mn[2] >= JB6.mn[2]: return 6
                if FE.mx[2] <= JB4.mx[2]: return 4
                return 5
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Yplus", "Yminus"): return err()
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
            if dd in ("Zplus", "Zminus"): return err()
        elif fe_n == "y":
            if dd == "Xplus": return 3
            if dd == "Xminus": return 1
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Yplus", "Yminus"): return err()
        elif fe_n == "z":
            if dd in ("Yplus", "Yminus"):
                if FE.mn[2] >= JB6.mn[2]: return 6
                if FE.mx[2] <= JB4.mx[2]: return 4
                return 5
            if dd == "Zplus": return 6
            if dd == "Zminus": return 4
            if dd in ("Xplus", "Xminus"): return err()
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
            if dd in ("Yplus", "Yminus"): return err()
        elif fe_n == "y":
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"):
                if FE.mn[1] >= JB3.mn[1]: return 3
                if FE.mx[1] <= JB1.mx[1]: return 1
                return 2
            if dd in ("Xplus", "Xminus"): return err()
        elif fe_n == "z":
            if dd == "Xplus": return 6
            if dd == "Xminus": return 4
            if dd == "Yplus": return 3
            if dd == "Yminus": return 1
            if dd in ("Zplus", "Zminus"): return err()
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
    b = elem.bbox
    small_axes = smallest_face_axes(b)

    # short: on one of four smallest faces
    for ax in small_axes:
        if is_on_face(b, ax, p_on_elem):
            return "short"

    # else on largest face; border if within 0.5m of in-plane edges
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


def build_cz_matrix(elements: List[ElementInfo], se_id: int) -> Dict[int, Dict[int, str]]:
    by_id = {e.ifc_id: e for e in elements}
    ids = [e.ifc_id for e in elements]
    cz: Dict[int, Dict[int, str]] = {i: {} for i in ids}

    # direct CZ for sufficiently contacting pairs
    for i in ids:
        for j in ids:
            if i == j:
                continue
            ei, ej = by_id[i], by_id[j]
            if not has_sufficient_contact(ej, ei):
                continue
            p = ei.bbox.clamp_point(ej.bbox.center())
            cz[i][j] = cz_for_element_at_contact(ei, p)

    # neutral 0 for flanker-flanker separated by SE
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
# Junction type rules (subset – extend as needed)
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

    # Tv1-2:4 + alias Tv1-2-4
    mk_rule("Tv1-2:4", ["n", "n", "o"], [
        (3, 1, "short"), (3, 2, "short"),
        (1, 3, "border"), (2, 3, "border"),
        (1, 2, "short"), (2, 1, "short"),
    ]),
    mk_rule("Tv1-2-4", ["n", "n", "o"], [
        (3, 1, "short"), (3, 2, "short"),
        (1, 3, "border"), (2, 3, "border"),
        (1, 2, "short"), (2, 1, "short"),
    ]),
]


def multiset(lst: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


def match_rule(elements: List[ElementInfo], se_id: int) -> Tuple[str, Dict[str, Any]]:
    assign_dir_labels(elements)
    cz = build_cz_matrix(elements, se_id)

    dbg = {
        "k": len(elements),
        "dirs": {e.ifc_id: e.dir_label for e in elements},
        "cz_pairs": {str(i): cz[i] for i in cz},
        "se_id": se_id,
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

        # candidates: all walls/slabs with sufficient (not edge-only) contact to SE
        candidates: List[ElementInfo] = []
        for fe_id, fe in infos.items():
            if fe_id == se_id:
                continue
            if not (is_wall(fe.ifc_type) or is_slab(fe.ifc_type)):
                continue
            if has_sufficient_contact(fe, se):
                candidates.append(fe)

        # JB assignment
        for fe in candidates:
            jb_id = assign_fe_to_jb(fe, se, jbs)
            if jb_id is None:
                continue
            if fe.ifc_id not in jbs[jb_id].fe_ids:
                jbs[jb_id].fe_ids.append(fe.ifc_id)

        # per-JB junctions
        for jb_id, jb in sorted(jbs.items(), key=lambda x: x[0]):
            fe_list = jb.fe_ids[:3]
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

        # COMBINED junction uses all close candidates (not only assigned via JB), pick closest 3
        if candidates:
            cand_sorted = sorted(candidates, key=lambda fe: se.bbox.distance_to(fe.bbox))
            elem_ids = [se_id] + [fe.ifc_id for fe in cand_sorted[:3]]
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

    # Deduplicate exact element sets, prefer recognized types
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
    ifc_path = "./ifc-models/model_Tv2-1-4.ifc" if len(sys.argv) < 2 else sys.argv[1]
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