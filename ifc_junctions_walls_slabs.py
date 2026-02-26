#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IFC Junction Extraction (Walls + Slabs) – extended for 3-element T junctions (Tv2-1-4 / Th2-1-4)
Key upgrades:
- Build BOTH: per-JB junctions AND combined junctions per separating element
- Pairwise connection zones CZ(e -> o) instead of single CZ per element
- Neutral CZ = "0" for flanker-flanker pairs when separated by the separating element
- Rule engine matches on directions + pairwise CZ constraints

Usage:
  python ifc_junctions_paircz.py            # uses ./model.ifc
  python ifc_junctions_paircz.py my.ifc

Outputs:
  junctions_raw.json       (debug: per-JB + combined)
  junctions_unique.json    (deduped junctions)
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any, FrozenSet

import numpy as np
import ifcopenshell
import ifcopenshell.geom


# -----------------------------
# Config
# -----------------------------

DIST_THRESH = 0.30     # close-to threshold for candidate filtering
PAD = 0.30
SPLIT = 0.50

BORDER_W = 0.50        # per your text/figures (0.5 m strips)
SHORT_W = 0.30

MAX_ELEMS_PER_JB = 4   # separating + up to 3 flanking
MAX_ELEMS_COMBINED = 4 # combined junction per separating element


Vec3 = Tuple[float, float, float]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class BBox:
    mn: Vec3
    mx: Vec3

    def intersects(self, other: "BBox") -> bool:
        return (
            self.mn[0] <= other.mx[0] and self.mx[0] >= other.mn[0] and
            self.mn[1] <= other.mx[1] and self.mx[1] >= other.mn[1] and
            self.mn[2] <= other.mx[2] and self.mx[2] >= other.mn[2]
        )

    def intersection(self, other: "BBox") -> Optional["BBox"]:
        mn = (max(self.mn[0], other.mn[0]), max(self.mn[1], other.mn[1]), max(self.mn[2], other.mn[2]))
        mx = (min(self.mx[0], other.mx[0]), min(self.mx[1], other.mx[1]), min(self.mx[2], other.mx[2]))
        if mn[0] <= mx[0] and mn[1] <= mx[1] and mn[2] <= mx[2]:
            return BBox(mn, mx)
        return None

    def center(self) -> Vec3:
        return ((self.mn[0] + self.mx[0]) / 2.0, (self.mn[1] + self.mx[1]) / 2.0, (self.mn[2] + self.mx[2]) / 2.0)

    def size(self) -> Vec3:
        return (self.mx[0] - self.mn[0], self.mx[1] - self.mn[1], self.mx[2] - self.mn[2])

    def distance_to(self, other: "BBox") -> float:
        dx = max(0.0, other.mn[0] - self.mx[0], self.mn[0] - other.mx[0])
        dy = max(0.0, other.mn[1] - self.mx[1], self.mn[1] - other.mx[1])
        dz = max(0.0, other.mn[2] - self.mx[2], self.mn[2] - other.mx[2])
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class ElementInfo:
    ifc_id: int
    guid: str
    ifc_type: str
    name: str
    bbox: BBox
    dir_label: str = ""   # n/m/o within a junction context
    dist: float = 0.0     # for debug


@dataclass
class JunctionBox:
    jb_id: int
    bbox: BBox
    elements: List[ElementInfo]


# -----------------------------
# IFC helpers
# -----------------------------

def create_settings():
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
# Relation-based candidate signals (best-effort)
# -----------------------------

def connected_elements_via_relconnects(model, se_id: int) -> Set[int]:
    ids = set()
    for r in model.by_type("IfcRelConnectsElements"):
        try:
            a = r.RelatingElement
            b = r.RelatedElement
            if not a or not b:
                continue
            if a.id() == se_id:
                ids.add(b.id())
            elif b.id() == se_id:
                ids.add(a.id())
        except Exception:
            pass
    return ids


def adjacent_via_spaceboundaries(model, se_id: int) -> Set[int]:
    ids = set()
    rels = model.by_type("IfcRelSpaceBoundary")
    for r in rels:
        try:
            e = r.RelatedBuildingElement
            if not e or e.id() != se_id:
                continue
            space = r.RelatingSpace
            if not space:
                continue
            for r2 in rels:
                try:
                    if r2.RelatingSpace and r2.RelatingSpace.id() == space.id():
                        e2 = r2.RelatedBuildingElement
                        if e2 and e2.id() != se_id:
                            ids.add(e2.id())
                except Exception:
                    pass
        except Exception:
            pass
    return ids


# -----------------------------
# Axes + directions (axis-aligned assumption)
# -----------------------------

def wall_len_axis_xy(b: BBox) -> int:
    sx, sy, _ = b.size()
    return 0 if sx >= sy else 1


def assign_nm_o_labels(elements: List[ElementInfo]) -> None:
    """
    n = first wall (reference)
    m = perpendicular wall(s)
    o = slabs
    """
    for e in elements:
        if is_slab(e.ifc_type):
            e.dir_label = "o"

    walls = [e for e in elements if is_wall(e.ifc_type)]
    if not walls:
        return

    ref = walls[0]
    ref.dir_label = "n"
    ref_ax = wall_len_axis_xy(ref.bbox)

    for w in walls[1:]:
        ax = wall_len_axis_xy(w.bbox)
        w.dir_label = "n" if ax == ref_ax else "m"


# -----------------------------
# Junction boxes
# -----------------------------

def build_junction_boxes_for_wall(se: ElementInfo) -> List[JunctionBox]:
    b = se.bbox
    mn = list(b.mn)
    mx = list(b.mx)
    len_ax = wall_len_axis_xy(b)
    thick_ax = 1 if len_ax == 0 else 0
    h_ax = 2
    mid_len = (mn[len_ax] + mx[len_ax]) / 2.0

    def mk(mn2, mx2): return BBox(tuple(mn2), tuple(mx2))

    boxes: List[JunctionBox] = []

    for jb_id, (a0, a1) in enumerate([(mn[len_ax] - PAD, mid_len + PAD),
                                     (mid_len - PAD, mx[len_ax] + PAD)], start=1):
        mn2, mx2 = mn.copy(), mx.copy()
        mn2[thick_ax] = mn[thick_ax] - SPLIT
        mx2[thick_ax] = mn[thick_ax] + SPLIT
        mn2[len_ax], mx2[len_ax] = a0, a1
        boxes.append(JunctionBox(jb_id, mk(mn2, mx2), elements=[se]))

    for jb_id, (a0, a1) in enumerate([(mn[len_ax] - PAD, mid_len + PAD),
                                     (mid_len - PAD, mx[len_ax] + PAD)], start=3):
        mn2, mx2 = mn.copy(), mx.copy()
        mn2[thick_ax] = mx[thick_ax] - SPLIT
        mx2[thick_ax] = mx[thick_ax] + SPLIT
        mn2[len_ax], mx2[len_ax] = a0, a1
        boxes.append(JunctionBox(jb_id, mk(mn2, mx2), elements=[se]))

    mn5, mx5 = mn.copy(), mx.copy()
    mn5[h_ax] = mn[h_ax] - DIST_THRESH
    mx5[h_ax] = mn[h_ax] + DIST_THRESH
    mn5[0] -= PAD; mn5[1] -= PAD
    mx5[0] += PAD; mx5[1] += PAD
    boxes.append(JunctionBox(5, mk(mn5, mx5), elements=[se]))

    mn6, mx6 = mn.copy(), mx.copy()
    mn6[h_ax] = mx[h_ax] - DIST_THRESH
    mx6[h_ax] = mx[h_ax] + DIST_THRESH
    mn6[0] -= PAD; mn6[1] -= PAD
    mx6[0] += PAD; mx6[1] += PAD
    boxes.append(JunctionBox(6, mk(mn6, mx6), elements=[se]))

    return boxes


def build_junction_boxes_for_slab(se: ElementInfo) -> List[JunctionBox]:
    b = se.bbox
    mn = list(b.mn)
    mx = list(b.mx)

    def mk(mn2, mx2): return BBox(tuple(mn2), tuple(mx2))
    boxes: List[JunctionBox] = []

    # perimeter bands (JB1..4)
    mnw, mxw = mn.copy(), mx.copy()
    mnw[0] = mn[0] - PAD
    mxw[0] = mn[0] + BORDER_W + PAD
    mnw[1] -= PAD; mxw[1] += PAD
    boxes.append(JunctionBox(1, mk(mnw, mxw), elements=[se]))

    mne, mxe = mn.copy(), mx.copy()
    mne[0] = mx[0] - BORDER_W - PAD
    mxe[0] = mx[0] + PAD
    mne[1] -= PAD; mxe[1] += PAD
    boxes.append(JunctionBox(2, mk(mne, mxe), elements=[se]))

    mns, mxs = mn.copy(), mx.copy()
    mns[1] = mn[1] - PAD
    mxs[1] = mn[1] + BORDER_W + PAD
    mns[0] -= PAD; mxs[0] += PAD
    boxes.append(JunctionBox(3, mk(mns, mxs), elements=[se]))

    mnn, mxn = mn.copy(), mx.copy()
    mnn[1] = mx[1] - BORDER_W - PAD
    mxn[1] = mx[1] + PAD
    mnn[0] -= PAD; mxn[0] += PAD
    boxes.append(JunctionBox(4, mk(mnn, mxn), elements=[se]))

    # below (JB5) and above (JB6)
    mnb, mxb = mn.copy(), mx.copy()
    mnb[2] = mn[2] - DIST_THRESH
    mxb[2] = mn[2] + DIST_THRESH
    mnb[0] -= PAD; mnb[1] -= PAD
    mxb[0] += PAD; mxb[1] += PAD
    boxes.append(JunctionBox(5, mk(mnb, mxb), elements=[se]))

    mna, mxa = mn.copy(), mx.copy()
    mna[2] = mx[2] - DIST_THRESH
    mxa[2] = mx[2] + DIST_THRESH
    mna[0] -= PAD; mna[1] -= PAD
    mxa[0] += PAD; mxa[1] += PAD
    boxes.append(JunctionBox(6, mk(mna, mxa), elements=[se]))

    return boxes


# -----------------------------
# Pairwise connection zones CZ(e->o) + neutral "0"
# -----------------------------

def classify_zone_on_element(elem: ElementInfo, contact_bbox: BBox) -> str:
    """
    Generic but stable:
    - short if contact centroid is near any bbox extreme face (within SHORT_W)
    - else border/middle on dominant face:
      * slab: border near x/y edges
      * wall: border near length edges or height edges
    """
    c = contact_bbox.center()
    mn, mx = elem.bbox.mn, elem.bbox.mx
    sx, sy, sz = elem.bbox.size()

    # short heuristic near any min/max face
    if (
        abs(c[0] - mn[0]) <= SHORT_W or abs(mx[0] - c[0]) <= SHORT_W or
        abs(c[1] - mn[1]) <= SHORT_W or abs(mx[1] - c[1]) <= SHORT_W or
        abs(c[2] - mn[2]) <= SHORT_W or abs(mx[2] - c[2]) <= SHORT_W
    ):
        return "short"

    if is_slab(elem.ifc_type):
        near_x = (c[0] <= mn[0] + BORDER_W) or (c[0] >= mx[0] - BORDER_W)
        near_y = (c[1] <= mn[1] + BORDER_W) or (c[1] >= mx[1] - BORDER_W)
        return "border" if (near_x or near_y) else "middle"

    # wall
    thick_ax = 0 if sx <= sy else 1
    len_ax = 1 if thick_ax == 0 else 0
    near_len = (c[len_ax] <= mn[len_ax] + BORDER_W) or (c[len_ax] >= mx[len_ax] - BORDER_W)
    near_h = (c[2] <= mn[2] + BORDER_W) or (c[2] >= mx[2] - BORDER_W)
    return "border" if (near_len or near_h) else "middle"


def compute_pair_cz(e: ElementInfo, o: ElementInfo) -> Optional[str]:
    """
    If AABBs intersect -> classify zone on e.
    If not -> None (handled by neutral logic elsewhere).
    """
    inter = e.bbox.intersection(o.bbox)
    if inter is None:
        return None
    return classify_zone_on_element(e, inter)


def build_cz_matrix(elements: List[ElementInfo], separating_id: int) -> Dict[int, Dict[int, str]]:
    """
    cz[e][o] for all ordered pairs e!=o.
    - if e intersects o -> short/border/middle
    - else if (e and o are BOTH flankers, and both intersect separator, and do NOT intersect each other)
      -> cz[e][o] = "0"  and cz[o][e] = "0"
    """
    ids = [e.ifc_id for e in elements]
    by_id = {e.ifc_id: e for e in elements}
    cz: Dict[int, Dict[int, str]] = {i: {} for i in ids}

    # first pass: actual contacts
    for i in ids:
        for j in ids:
            if i == j:
                continue
            v = compute_pair_cz(by_id[i], by_id[j])
            if v is not None:
                cz[i][j] = v

    # neutral "0" for flanker-flanker separated by separator
    sep = by_id[separating_id]
    flanker_ids = [i for i in ids if i != separating_id]

    for a in flanker_ids:
        for b in flanker_ids:
            if a >= b:
                continue
            # must not intersect directly
            if by_id[a].bbox.intersects(by_id[b].bbox):
                continue
            # but both must intersect separator
            if not by_id[a].bbox.intersects(sep.bbox):
                continue
            if not by_id[b].bbox.intersects(sep.bbox):
                continue
            cz[a][b] = "0"
            cz[b][a] = "0"

    return cz


# -----------------------------
# Rule engine with pairwise CZ constraints
# -----------------------------

# We map local indices 1..k (after sorting by dir n,m,o) to actual ids
# Rule format:

RULES = [
    # -------------------------
    # 2-Element Junctions (k=2)  --- ROBUST ---
    # -------------------------

    # Lh1-2 (Wall-Wall L)
    # Paper: n short, m border
    # Robust: m->n may appear as short depending on bbox contact -> allow short|border
    {
        "type": "Lh1-2",
        "k": 2,
        "dir": {1: "n", 2: "m"},
        "cz": [
            (1, 2, "short"),
            (2, 1, {"border", "short"}),
        ],
    },

    # Lv1-2 (Wall-Slab L)
    # Paper: n short, o border
    # Robust: slab->wall sometimes becomes short (bbox contact at edge) -> allow short|border
    {
        "type": "Lv1-2",
        "k": 2,
        "dir": {1: "n", 2: "o"},
        "cz": [
            (1, 2, "short"),
            (2, 1, {"border", "short"}),
        ],
    },

    # Tv2-13
    # Paper: n short, o middle
    # Robust: o->n can appear border/middle depending on your slab border width and contact centroid -> allow middle|border
    {
        "type": "Tv2-13",
        "k": 2,
        "dir": {1: "n", 2: "o"},
        "cz": [
            (1, 2, "short"),
            (2, 1, {"middle", "border"}),
        ],
    },

    # Th1-24
    # Paper: n short, m middle
    # Robust: m->n can drift to border/middle -> allow middle|border
    {
        "type": "Th1-24",
        "k": 2,
        "dir": {1: "n", 2: "m"},
        "cz": [
            (1, 2, "short"),
            (2, 1, {"middle", "border"}),
        ],
    },

    # Tv1-24
    # Paper: n middle, o short
    # Robust: n->o sometimes border/middle -> allow middle|border
    {
        "type": "Tv1-24",
        "k": 2,
        "dir": {1: "n", 2: "o"},
        "cz": [
            (1, 2, {"middle", "border"}),
            (2, 1, "short"),
        ],
    },

    # -------------------------
    # 3-Element Junctions (k=3)
    # -------------------------

    # Th2-1-4  (dein bestätigter korrekter Fall)
    {
        "type": "Th2-1-4",
        "k": 3,
        "dir": {1: "n", 2: "m", 3: "m"},
        "cz": [
            (1, 2, "border"),
            (1, 3, "border"),
            (2, 1, "short"),
            (3, 1, "short"),
            (2, 3, "0"),
            (3, 2, "0"),
        ],
    },

    # Tv2-1-4  (dein funktionierender Fall: n,n,o + n<->n = 0 + n->o short)
    {
        "type": "Tv2-1-4",
        "k": 3,
        "dir": {1: "n", 2: "n", 3: "o"},
        "cz": [
            (1, 3, "short"),
            (2, 3, "short"),
            (1, 2, "0"),
            (2, 1, "0"),
            # optional streng:
            # (3, 1, {"short", "border"}),
            # (3, 2, {"short", "border"}),
        ],
    },

    # OPTIONAL: wenn du Tv1-2:4 brauchst (aus deiner Abb. 5.43)
    # o short short; n border short; n border short
    # => o->n short, n->o border|short; n<->n short
    {
        "type": "Tv1-2:4",
        "k": 3,
        "dir": {1: "n", 2: "n", 3: "o"},
        "cz": [
            (3, 1, "short"),
            (3, 2, "short"),
            (1, 3, {"border", "short"}),
            (2, 3, {"border", "short"}),
            (1, 2, {"short", "border"}),  # je nach Wand/Wand-Lage
            (2, 1, {"short", "border"}),
        ],
    },
]


def match_rules(elements: List[ElementInfo], separating_id: int) -> Tuple[str, Dict[str, Any]]:
    """
    Sort local order: n, m, o; then by id.
    Compute CZ matrix and test rules.
    """
    def sort_key(e: ElementInfo):
        order = {"n": 0, "m": 1, "o": 2, "": 9}
        return (order.get(e.dir_label, 9), e.ifc_id)

    elems_sorted = sorted(elements, key=sort_key)
    idx_to_id = {i + 1: elems_sorted[i].ifc_id for i in range(len(elems_sorted))}
    id_to_idx = {v: k for k, v in idx_to_id.items()}

    cz = build_cz_matrix(elements, separating_id)

    dbg = {
        "k": len(elements),
        "local_order": {i: idx_to_id[i] for i in idx_to_id},
        "dirs": {e.ifc_id: e.dir_label for e in elements},
        "cz_pairs": {str(k): cz[k] for k in cz},
        "separating_id": separating_id,
    }

    for rule in RULES:
        if rule["k"] != len(elements):
            continue

        # check dirs
        ok = True
        for idx, d in rule["dir"].items():
            eid = idx_to_id.get(idx)
            if eid is None:
                ok = False
                break
            if next(e for e in elements if e.ifc_id == eid).dir_label != d:
                ok = False
                break
        if not ok:
            continue

        # check pairwise CZ
        for (a_idx, b_idx, want) in rule["cz"]:
            a_id = idx_to_id.get(a_idx)
            b_id = idx_to_id.get(b_idx)
            if a_id is None or b_id is None:
                ok = False
                break
            got = cz.get(a_id, {}).get(b_id)
            # want can be:
            # - str: exact match
            # - set/list/tuple: any-of match
            if isinstance(want, (set, list, tuple)):
                if got not in want:
                    ok = False
                    break
            else:
                if got != want:
                    ok = False
                    break

        if ok:
            return rule["type"], dbg

    return "UNKNOWN", dbg


# -----------------------------
# Dedup
# -----------------------------

def _type_score(t: str) -> int:
    if t == "NONE":
        return 0
    if t == "UNKNOWN":
        return 1
    return 2


def dedupe_unique(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[FrozenSet[int], Dict[str, Any]] = {}
    for r in rows:
        key = frozenset(r["element_ids"])
        if key not in best or _type_score(r["junction_type"]) > _type_score(best[key]["junction_type"]):
            best[key] = r
    out = [v for v in best.values() if v["junction_type"] != "NONE"]
    out.sort(key=lambda x: (x["junction_type"], x["element_ids"]))
    return out


# -----------------------------
# Main analysis
# -----------------------------

def analyze_ifc(ifc_path: str,
                out_raw_json: str = "junctions_raw.json",
                out_unique_json: str = "junctions_unique.json") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    model = ifcopenshell.open(ifc_path)
    settings = create_settings()

    raw = collect_walls_slabs(model)

    infos: Dict[int, ElementInfo] = {}
    for e in raw:
        bb = element_bbox(settings, e)
        if bb is None:
            continue
        infos[e.id()] = ElementInfo(
            ifc_id=e.id(),
            guid=getattr(e, "GlobalId", "") or "",
            ifc_type=e.is_a(),
            name=getattr(e, "Name", "") or "",
            bbox=bb,
        )

    separating_ids = [eid for eid, inf in infos.items() if is_wall(inf.ifc_type) or is_slab(inf.ifc_type)]

    rows: List[Dict[str, Any]] = []

    for se_id in separating_ids:
        se = infos[se_id]

        # candidates: relations + distance to all walls/slabs (fallback)
        rel_ids = connected_elements_via_relconnects(model, se_id)
        sb_ids = adjacent_via_spaceboundaries(model, se_id)
        candidate_ids = (rel_ids | sb_ids)

        # fallback: if relations empty, consider all walls/slabs
        if not candidate_ids:
            candidate_ids = set(infos.keys())

        candidate_ids.discard(se_id)
        candidate_ids = {cid for cid in candidate_ids if cid in infos and (is_wall(infos[cid].ifc_type) or is_slab(infos[cid].ifc_type))}

        # distance filter
        flankers = []
        for cid in candidate_ids:
            d = se.bbox.distance_to(infos[cid].bbox)
            if d <= DIST_THRESH:
                fe = infos[cid]
                fe.dist = d
                flankers.append(fe)

        # junction boxes
        jbs = build_junction_boxes_for_wall(se) if is_wall(se.ifc_type) else build_junction_boxes_for_slab(se)

        # assign flankers to best JB
        for fe in flankers:
            best_jb = None
            best_score = float("inf")
            fe_center = np.array(fe.bbox.center(), dtype=float)
            for jb in jbs:
                if jb.bbox.intersects(fe.bbox):
                    jb_center = np.array(jb.bbox.center(), dtype=float)
                    score = float(np.linalg.norm(jb_center - fe_center))
                    if score < best_score:
                        best_score = score
                        best_jb = jb
            if best_jb is not None and len(best_jb.elements) < MAX_ELEMS_PER_JB:
                best_jb.elements.append(fe)

        # --- (A) per-JB rows ---
        for jb in jbs:
            elems = jb.elements[:]
            if len(elems) <= 1:
                rows.append({
                    "element_ids": [se_id],
                    "junction_type": "NONE",
                    "junction_scope": f"JB{jb.jb_id}",
                    "debug": {"reason": "no flanking"},
                })
                continue

            assign_nm_o_labels(elems)
            jtype, dbg = match_rules(elems, separating_id=se_id)

            rows.append({
                "element_ids": sorted([e.ifc_id for e in elems]),
                "junction_type": jtype,
                "junction_scope": f"JB{jb.jb_id}",
                "elements": [{
                    "ifc_id": e.ifc_id,
                    "guid": e.guid,
                    "type": e.ifc_type,
                    "name": e.name,
                    "dir": e.dir_label,
                    "dist": e.dist,
                } for e in elems],
                "debug": dbg,
            })

        # --- (B) combined junction per separating element ---
        # collect all flankers assigned anywhere to any JB
        combined_ids: Set[int] = set([se_id])
        for jb in jbs:
            for e in jb.elements:
                combined_ids.add(e.ifc_id)

        if len(combined_ids) >= 3:
            # keep up to MAX_ELEMS_COMBINED: se + nearest flankers
            fl_sorted = sorted([infos[i] for i in combined_ids if i != se_id], key=lambda x: x.dist)
            keep = [se] + fl_sorted[: (MAX_ELEMS_COMBINED - 1)]
            elems = keep

            assign_nm_o_labels(elems)
            jtype, dbg = match_rules(elems, separating_id=se_id)

            rows.append({
                "element_ids": sorted([e.ifc_id for e in elems]),
                "junction_type": jtype,
                "junction_scope": "COMBINED",
                "elements": [{
                    "ifc_id": e.ifc_id,
                    "guid": e.guid,
                    "type": e.ifc_type,
                    "name": e.name,
                    "dir": e.dir_label,
                    "dist": e.dist,
                } for e in elems],
                "debug": dbg,
            })

    unique = dedupe_unique(rows)

    with open(out_raw_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(out_unique_json, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    return unique, rows


def main():
    ifc_path = "./ifc-models/model_Tv2-1-4.ifc" if len(sys.argv) < 2 else sys.argv[1]
    if not os.path.exists(ifc_path):
        print(f"ERROR: IFC file not found: {ifc_path}")
        sys.exit(1)

    unique, rows = analyze_ifc(ifc_path)
    print("Done.")
    print(f"- Raw:    {len(rows)} -> junctions_raw.json")
    print(f"- Unique: {len(unique)} -> junctions_unique.json")


if __name__ == "__main__":
    main()