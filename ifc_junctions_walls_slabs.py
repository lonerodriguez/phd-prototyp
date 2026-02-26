#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IFC Junction Extraction (Walls + Slabs) – paper-inspired pipeline

Reads an IFC, considers ONLY:
- Walls: IfcWall, IfcWallStandardCase
- Slabs/Ceilings: IfcSlab (and optionally IfcRoof)

Workflow:
1) Read IFC + geometry (AABB)
2) Filter flanking candidates using:
   - IfcRelConnectsElements
   - IfcRelSpaceBoundary (best-effort)
   - Storey neighborhood (same + above/below) (best-effort)
   - Distance threshold (<= 0.30 m)
3) Build Junction Boxes (JB) around each separating element:
   - Wall: 6 boxes (4 side boxes split along length + below + above)
   - Slab: 6 boxes (4 perimeter bands + below + above)
4) Assign flanking elements to the best matching JB (intersection + closest center)
5) Derive junction type with a small rule set:
   - Robust: Wall+Wall, perpendicular (dir='m') and cz='short' => Lh1-2
   - Plus the excerpted 2-element rules used earlier
   - Everything else => UNKNOWN_* placeholders (extend here with full 15-type table)

Usage (IFC in same folder):
    python ifc_junctions_walls_slabs.py model.ifc
Optional:
    python ifc_junctions_walls_slabs.py model.ifc out.json

Dependencies:
    pip install ifcopenshell numpy
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import ifcopenshell
import ifcopenshell.geom


# -----------------------------
# Config (paper-inspired defaults)
# -----------------------------

DIST_THRESH = 0.30   # max AABB distance to consider flanking [m]
PAD = 0.30           # padding around bboxes [m]
SPLIT = 0.50         # depth/width for wall side boxes [m]
BORDER_W = 0.30      # border zone thickness [m]
SHORT_W = 0.30       # "short" zone near wall ends [m]

MAX_ELEMS_PER_JB = 4  # separating + up to 3 flanking (paper mentions max 4 elements)


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

    def center(self) -> Vec3:
        return (
            (self.mn[0] + self.mx[0]) / 2.0,
            (self.mn[1] + self.mx[1]) / 2.0,
            (self.mn[2] + self.mx[2]) / 2.0,
        )

    def size(self) -> Vec3:
        return (
            self.mx[0] - self.mn[0],
            self.mx[1] - self.mn[1],
            self.mx[2] - self.mn[2],
        )

    def distance_to(self, other: "BBox") -> float:
        """Minimum distance between two AABBs (0 if intersect)."""
        dx = max(0.0, other.mn[0] - self.mx[0], self.mn[0] - other.mx[0])
        dy = max(0.0, other.mn[1] - self.mx[1], self.mn[1] - other.mx[1])
        dz = max(0.0, other.mn[2] - self.mx[2], self.mn[2] - other.mx[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass
class ElementInfo:
    ifc_id: int
    guid: str
    ifc_type: str
    name: str
    bbox: BBox
    # Paper abstraction direction labels relative to separating element:
    # n = along separating element main horizontal direction
    # m = perpendicular horizontal direction
    # o = vertical (slab/wall-to-slab)
    dir_label: str = ""
    dd_label: str = ""   # distance-direction label (heuristic)
    dist: float = 0.0


@dataclass
class JunctionBox:
    jb_id: int
    bbox: BBox
    elements: List[ElementInfo]  # includes separating element + flanking assigned


# -----------------------------
# IFC helpers
# -----------------------------

def create_settings():
    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, True)
    return s


def element_bbox(settings, element) -> Optional[BBox]:
    """Compute AABB from tessellated geometry; return None if geometry fails."""
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape((-1, 3))
        mn = tuple(np.min(verts, axis=0).tolist())
        mx = tuple(np.max(verts, axis=0).tolist())
        return BBox(mn, mx)
    except Exception:
        return None


def is_wall(t: str) -> bool:
    t = t.lower()
    return t.startswith("ifcwall")  # includes IfcWallStandardCase


def is_slab(t: str) -> bool:
    t = t.lower()
    return t.startswith("ifcslab") or t.startswith("ifcroof")


def collect_walls_slabs(model) -> List:
    els = []
    els.extend(model.by_type("IfcWall"))
    els.extend(model.by_type("IfcWallStandardCase"))
    els.extend(model.by_type("IfcSlab"))
    # optional: some exports classify ceilings as roofs
    els.extend(model.by_type("IfcRoof"))
    out, seen = [], set()
    for e in els:
        if e.id() not in seen:
            out.append(e)
            seen.add(e.id())
    return out


# -----------------------------
# Semantic filtering (relations)
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
    """Best-effort: collect other elements bounding the same space as se via IfcRelSpaceBoundary."""
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


def storey_of_element(model, element) -> Optional[int]:
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        try:
            if element in rel.RelatedElements:
                container = rel.RelatingStructure
                if container and container.is_a("IfcBuildingStorey"):
                    return container.id()
        except Exception:
            pass
    return None


def elements_in_storey(model, storey_id: int) -> List:
    out = []
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        try:
            if rel.RelatingStructure and rel.RelatingStructure.id() == storey_id:
                out.extend(list(rel.RelatedElements))
        except Exception:
            pass
    return out


def neighbor_storeys(model, storey_id: int) -> Tuple[Optional[int], Optional[int]]:
    storeys = model.by_type("IfcBuildingStorey")
    lst = []
    for s in storeys:
        elev = getattr(s, "Elevation", None)
        if elev is None:
            continue
        lst.append((s.id(), float(elev)))
    lst.sort(key=lambda x: x[1])
    ids = [i for i, _ in lst]
    if storey_id not in ids:
        return None, None
    idx = ids.index(storey_id)
    below = ids[idx - 1] if idx > 0 else None
    above = ids[idx + 1] if idx < len(ids) - 1 else None
    return below, above


# -----------------------------
# Axes + directions + zones (FIXED)
# -----------------------------

def wall_axes_from_bbox(b: BBox) -> Tuple[int, int, int]:
    """
    Wall-specific axes:
    - length axis: larger of X/Y
    - thickness axis: smaller of X/Y
    - height axis: Z (=2)
    """
    sx, sy, _ = b.size()
    if sx >= sy:
        len_ax, thick_ax = 0, 1
    else:
        len_ax, thick_ax = 1, 0
    return len_ax, thick_ax, 2


def slab_axes_from_bbox(b: BBox) -> Tuple[int, int, int]:
    """
    Slab-specific axes (plan):
    - long axis: larger of X/Y
    - mid axis: smaller of X/Y
    - vertical: Z (=2)
    """
    sx, sy, _ = b.size()
    if sx >= sy:
        long_ax, mid_ax = 0, 1
    else:
        long_ax, mid_ax = 1, 0
    return long_ax, mid_ax, 2


def element_dir_label(se: ElementInfo, fe: ElementInfo) -> str:
    # slabs/ceilings treated as vertical category when flanking
    if is_slab(fe.ifc_type):
        return "o"

    # wall-wall: compare wall length axes in plan (XY) => perpendicular => "m"
    if is_wall(se.ifc_type) and is_wall(fe.ifc_type):
        se_len, _, _ = wall_axes_from_bbox(se.bbox)
        fe_len, _, _ = wall_axes_from_bbox(fe.bbox)
        return "n" if fe_len == se_len else "m"

    # slab-wall: compare wall length axis to slab long axis in plan
    if is_slab(se.ifc_type) and is_wall(fe.ifc_type):
        se_long, _, _ = slab_axes_from_bbox(se.bbox)
        fe_len, _, _ = wall_axes_from_bbox(fe.bbox)
        return "n" if fe_len == se_long else "m"

    return "n"


def distance_direction_label(se: ElementInfo, fe: ElementInfo) -> str:
    """
    Which axis explains separation most: map to n/m/o using SE axes.
    """
    gaps = []
    for ax in range(3):
        gap = max(0.0, fe.bbox.mn[ax] - se.bbox.mx[ax], se.bbox.mn[ax] - fe.bbox.mx[ax])
        gaps.append(gap)
    max_ax = int(np.argmax(gaps))

    if max_ax == 2:
        return "o"

    if is_wall(se.ifc_type):
        se_len, _, _ = wall_axes_from_bbox(se.bbox)
        return "n" if max_ax == se_len else "m"

    se_long, _, _ = slab_axes_from_bbox(se.bbox)
    return "n" if max_ax == se_long else "m"


def wall_connection_zone(se_wall_bbox: BBox, fe_bbox: BBox) -> str:
    """
    Zone on wall face (heuristic): short / border / middle.
    - short: near wall ends along length axis
    - border: near perimeter (ends or top/bottom)
    - middle: otherwise
    """
    len_ax, _, h_ax = wall_axes_from_bbox(se_wall_bbox)

    c = fe_bbox.center()
    proj_len = c[len_ax]
    proj_h = c[h_ax]

    if proj_len <= se_wall_bbox.mn[len_ax] + SHORT_W or proj_len >= se_wall_bbox.mx[len_ax] - SHORT_W:
        return "short"

    if (proj_len <= se_wall_bbox.mn[len_ax] + BORDER_W or proj_len >= se_wall_bbox.mx[len_ax] - BORDER_W or
        proj_h <= se_wall_bbox.mn[h_ax] + BORDER_W or proj_h >= se_wall_bbox.mx[h_ax] - BORDER_W):
        return "border"

    return "middle"


def slab_connection_zone(se_slab_bbox: BBox, fe_bbox: BBox) -> str:
    """
    Slab zone (heuristic): border / middle in plan (XY).
    """
    c = fe_bbox.center()
    x, y = c[0], c[1]
    near_x_edge = (x <= se_slab_bbox.mn[0] + BORDER_W) or (x >= se_slab_bbox.mx[0] - BORDER_W)
    near_y_edge = (y <= se_slab_bbox.mn[1] + BORDER_W) or (y >= se_slab_bbox.mx[1] - BORDER_W)
    return "border" if (near_x_edge or near_y_edge) else "middle"


# -----------------------------
# Junction boxes
# -----------------------------

def build_junction_boxes_for_wall(se: ElementInfo) -> List[JunctionBox]:
    """
    6 boxes around wall AABB:
    - 4 side boxes: two on each side of thickness, split into two halves along wall length
    - 1 below + 1 above
    """
    b = se.bbox
    mn = list(b.mn)
    mx = list(b.mx)

    len_ax, thick_ax, h_ax = wall_axes_from_bbox(b)
    mid_len = (mn[len_ax] + mx[len_ax]) / 2.0

    def mk(mn2, mx2) -> BBox:
        return BBox(tuple(mn2), tuple(mx2))

    boxes: List[JunctionBox] = []

    # side negative (thick-): JB1/JB2
    for jb_id, (a0, a1) in enumerate([(mn[len_ax] - PAD, mid_len + PAD),
                                     (mid_len - PAD, mx[len_ax] + PAD)], start=1):
        mn2, mx2 = mn.copy(), mx.copy()
        mn2[thick_ax] = mn[thick_ax] - SPLIT
        mx2[thick_ax] = mn[thick_ax] + SPLIT
        mn2[len_ax], mx2[len_ax] = a0, a1
        boxes.append(JunctionBox(jb_id, mk(mn2, mx2), elements=[se]))

    # side positive (thick+): JB3/JB4
    for jb_id, (a0, a1) in enumerate([(mn[len_ax] - PAD, mid_len + PAD),
                                     (mid_len - PAD, mx[len_ax] + PAD)], start=3):
        mn2, mx2 = mn.copy(), mx.copy()
        mn2[thick_ax] = mx[thick_ax] - SPLIT
        mx2[thick_ax] = mx[thick_ax] + SPLIT
        mn2[len_ax], mx2[len_ax] = a0, a1
        boxes.append(JunctionBox(jb_id, mk(mn2, mx2), elements=[se]))

    # below: JB5
    mn5, mx5 = mn.copy(), mx.copy()
    mn5[h_ax] = mn[h_ax] - DIST_THRESH
    mx5[h_ax] = mn[h_ax] + DIST_THRESH
    mn5[0] -= PAD; mn5[1] -= PAD
    mx5[0] += PAD; mx5[1] += PAD
    boxes.append(JunctionBox(5, mk(mn5, mx5), elements=[se]))

    # above: JB6
    mn6, mx6 = mn.copy(), mx.copy()
    mn6[h_ax] = mx[h_ax] - DIST_THRESH
    mx6[h_ax] = mx[h_ax] + DIST_THRESH
    mn6[0] -= PAD; mn6[1] -= PAD
    mx6[0] += PAD; mx6[1] += PAD
    boxes.append(JunctionBox(6, mk(mn6, mx6), elements=[se]))

    return boxes


def build_junction_boxes_for_slab(se: ElementInfo) -> List[JunctionBox]:
    """
    Slab junction boxes:
    - 4 perimeter bands in plan (W/E/S/N)
    - plus 1 below and 1 above
    """
    b = se.bbox
    mn = list(b.mn)
    mx = list(b.mx)

    def mk(mn2, mx2) -> BBox:
        return BBox(tuple(mn2), tuple(mx2))

    boxes: List[JunctionBox] = []

    # West band (JB1)
    mnw, mxw = mn.copy(), mx.copy()
    mnw[0] = mn[0] - PAD
    mxw[0] = mn[0] + BORDER_W + PAD
    mnw[1] -= PAD; mxw[1] += PAD
    boxes.append(JunctionBox(1, mk(mnw, mxw), elements=[se]))

    # East band (JB2)
    mne, mxe = mn.copy(), mx.copy()
    mne[0] = mx[0] - BORDER_W - PAD
    mxe[0] = mx[0] + PAD
    mne[1] -= PAD; mxe[1] += PAD
    boxes.append(JunctionBox(2, mk(mne, mxe), elements=[se]))

    # South band (JB3)
    mns, mxs = mn.copy(), mx.copy()
    mns[1] = mn[1] - PAD
    mxs[1] = mn[1] + BORDER_W + PAD
    mns[0] -= PAD; mxs[0] += PAD
    boxes.append(JunctionBox(3, mk(mns, mxs), elements=[se]))

    # North band (JB4)
    mnn, mxn = mn.copy(), mx.copy()
    mnn[1] = mx[1] - BORDER_W - PAD
    mxn[1] = mx[1] + PAD
    mnn[0] -= PAD; mxn[0] += PAD
    boxes.append(JunctionBox(4, mk(mnn, mxn), elements=[se]))

    # Below (JB5)
    mnb, mxb = mn.copy(), mx.copy()
    mnb[2] = mn[2] - DIST_THRESH
    mxb[2] = mn[2] + DIST_THRESH
    mnb[0] -= PAD; mnb[1] -= PAD
    mxb[0] += PAD; mxb[1] += PAD
    boxes.append(JunctionBox(5, mk(mnb, mxb), elements=[se]))

    # Above (JB6)
    mna, mxa = mn.copy(), mx.copy()
    mna[2] = mx[2] - DIST_THRESH
    mxa[2] = mx[2] + DIST_THRESH
    mna[0] -= PAD; mna[1] -= PAD
    mxa[0] += PAD; mxa[1] += PAD
    boxes.append(JunctionBox(6, mk(mna, mxa), elements=[se]))

    return boxes


# -----------------------------
# Rule engine (minimal but fixes your L case)
# -----------------------------

def derive_junction_type(se: ElementInfo, flanking: List[ElementInfo]) -> Tuple[str, Dict]:
    """
    Returns (junction_type, debug_dict).
    Implemented:
    - Robust L-junction: Wall+Wall perpendicular + cz=short => Lh1-2
    - Earlier excerpted 2-element wall rules retained
    Extend here for full paper rule table (3/4 elements, slabs etc.).
    """
    dbg = {"se_type": se.ifc_type, "flanking_count": len(flanking)}

    if len(flanking) == 0:
        return "NONE", dbg

    if len(flanking) == 1:
        fe = flanking[0]
        dbg.update({"fe_dir": fe.dir_label, "fe_dd": fe.dd_label})

        if is_wall(se.ifc_type):
            cz = wall_connection_zone(se.bbox, fe.bbox)
        else:
            cz = slab_connection_zone(se.bbox, fe.bbox)
        dbg["cz"] = cz

        # --- FIX: your expected case ---
        # Two walls forming an L: FE perpendicular to SE (dir='m') and connection at end zone => Lh1-2
        if is_wall(se.ifc_type) and is_wall(fe.ifc_type):
            if fe.dir_label == "m" and cz == "short":
                return "Lh1-2", dbg

            # Keep stricter/excerpted rules as fallback
            if fe.dir_label == "m" and fe.dd_label == "n" and cz == "short":
                return "Lh1-2", dbg
            if fe.dir_label == "m" and fe.dd_label == "m" and cz == "border":
                return "Lh1-2", dbg
            if fe.dir_label == "m" and fe.dd_label == "m" and cz == "middle":
                return "Th1-24", dbg

        # Generic placeholders for other 2-element combos
        if is_slab(se.ifc_type) and is_wall(fe.ifc_type):
            return "SLAB_WITH_WALL", dbg
        if is_wall(se.ifc_type) and is_slab(fe.ifc_type):
            return "WALL_WITH_SLAB", dbg

        return "UNKNOWN_2E", dbg

    dbg["flanking_dirs"] = [f.dir_label for f in flanking]
    dbg["flanking_dds"] = [f.dd_label for f in flanking]
    return "UNMAPPED_COMPLEX", dbg


# -----------------------------
# Main pipeline
# -----------------------------

def analyze_ifc(ifc_path: str, out_json: str = "junctions.json") -> List[Dict]:
    model = ifcopenshell.open(ifc_path)
    settings = create_settings()

    raw = collect_walls_slabs(model)

    # Precompute geometry for walls+slabs only
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

    # storey map (best effort)
    elem_storey: Dict[int, Optional[int]] = {}
    for eid in separating_ids:
        elem_storey[eid] = storey_of_element(model, model.by_id(eid))

    out_rows: List[Dict] = []

    for se_id in separating_ids:
        se = infos[se_id]

        # candidate IDs from:
        rel_ids = connected_elements_via_relconnects(model, se_id)
        sb_ids = adjacent_via_spaceboundaries(model, se_id)

        storey_ids: Set[int] = set()
        se_storey = elem_storey.get(se_id)
        if se_storey is not None:
            storey_ids.add(se_storey)
            below, above = neighbor_storeys(model, se_storey)
            if below:
                storey_ids.add(below)
            if above:
                storey_ids.add(above)

        storey_elem_ids: Set[int] = set()
        for sid in storey_ids:
            for e in elements_in_storey(model, sid):
                storey_elem_ids.add(e.id())

        candidate_ids = (rel_ids | sb_ids | storey_elem_ids)
        candidate_ids.discard(se_id)

        # keep only walls+slabs and those with geometry
        candidate_ids = {
            cid for cid in candidate_ids
            if cid in infos and (is_wall(infos[cid].ifc_type) or is_slab(infos[cid].ifc_type))
        }

        # distance filter
        filtered_ids = []
        for cid in candidate_ids:
            d = se.bbox.distance_to(infos[cid].bbox)
            if d <= DIST_THRESH:
                filtered_ids.append(cid)

        # build junction boxes depending on separating type
        if is_wall(se.ifc_type):
            jbs = build_junction_boxes_for_wall(se)
        else:
            jbs = build_junction_boxes_for_slab(se)

        # prepare flanking infos with direction + dd labels
        fl_infos: List[ElementInfo] = []
        for fid in filtered_ids:
            fe = infos[fid]
            fe.dist = se.bbox.distance_to(fe.bbox)
            fe.dir_label = element_dir_label(se, fe)
            fe.dd_label = distance_direction_label(se, fe)
            fl_infos.append(fe)

        # assign each FE to best intersecting JB
        for fe in fl_infos:
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

        # derive junction type per JB
        for jb in jbs:
            flanking = [e for e in jb.elements if e.ifc_id != se.ifc_id]
            jtype, dbg = derive_junction_type(se, flanking)

            out_rows.append({
                "separating": {
                    "ifc_id": se.ifc_id,
                    "guid": se.guid,
                    "type": se.ifc_type,
                    "name": se.name,
                    "bbox": {"mn": se.bbox.mn, "mx": se.bbox.mx},
                },
                "junction_box": jb.jb_id,
                "flanking": [{
                    "ifc_id": f.ifc_id,
                    "guid": f.guid,
                    "type": f.ifc_type,
                    "name": f.name,
                    "dir": f.dir_label,
                    "dd": f.dd_label,
                    "dist": f.dist
                } for f in flanking],
                "junction_type": jtype,
                "debug": dbg
            })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    return out_rows


def main():
    # If no args: default to model.ifc in current folder
    if len(sys.argv) < 2:
        ifc_path = "model.ifc"
    else:
        ifc_path = sys.argv[1]

    out_json = sys.argv[2] if len(sys.argv) > 2 else "junctions.json"
    rows = analyze_ifc(ifc_path, out_json=out_json)
    print(f"Done. Wrote {len(rows)} junction records to {out_json}")


if __name__ == "__main__":
    main()