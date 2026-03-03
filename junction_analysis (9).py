"""
=============================================================================
IFC Junction Analysis – Rotationsrobuste Stoßstellenanalyse
=============================================================================
Basiert auf: Châteauvieux-Hellwig et al. (2020/2022)
             "Analysis of the early-design timber models for sound insulation"

Anforderungen:
    pip install ifcopenshell numpy

Ausführung:
    python junction_analysis.py --ifc mein_modell.ifc --out ./output

Ausgabe:
    output/debug_junction_analysis.json   – alle Zwischenschritte
    output/junctions_result.json          – gefundene Stoßstellen
    output/junction_analysis.log          – vollständiges Log
=============================================================================
"""

import argparse
import datetime
import json
import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import norm

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.placement
    import ifcopenshell.util.element
except ImportError:
    print("[FEHLER] ifcopenshell nicht installiert.  ->  pip install ifcopenshell")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Globale Konstanten
# ---------------------------------------------------------------------------
MAX_DISTANCE     = 0.30
JB_OFFSET_THIN   = 0.30
JB_OFFSET_BORDER = 0.50
ANGLE_TOL        = 0.85   # cos(≈32°)
GEOM_TOL         = 1e-3

RELEVANT_IFC_TYPES = [
    "IfcWall", "IfcWallStandardCase",
    "IfcSlab",
    "IfcCurtainWall",
    "IfcMember",
    "IfcPlate",
]

log = logging.getLogger("junction_analysis")


# ===========================================================================
# DATENSTRUKTUREN
# ===========================================================================

@dataclass
class BoundingBox:
    min_pt: np.ndarray
    max_pt: np.ndarray

    def center(self) -> np.ndarray:
        return (self.min_pt + self.max_pt) * 0.5

    def size(self) -> np.ndarray:
        return self.max_pt - self.min_pt

    def to_dict(self) -> dict:
        return {"min": self.min_pt.tolist(), "max": self.max_pt.tolist()}


@dataclass
class BuildingElement:
    ifc_id:         int
    ifc_type:       str
    global_id:      str
    name:           str
    bbox:           BoundingBox
    n_vec:          np.ndarray
    u_vec:          np.ndarray
    v_vec:          np.ndarray
    storey_id:      int = -1
    storey_name:    str = ""
    elem_direction: str = ""
    distance_to_se: float = np.inf
    dd_local:       np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_vec_world:    np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_dict(self) -> dict:
        return {
            "ifc_id":    self.ifc_id,
            "ifc_type":  self.ifc_type,
            "global_id": self.global_id,
            "name":      self.name,
            "storey":    self.storey_name,
            "bbox":      self.bbox.to_dict(),
            "n_vec":     self.n_vec.tolist(),
            "u_vec":     self.u_vec.tolist(),
            "v_vec":     self.v_vec.tolist(),
            "elem_dir":  self.elem_direction,
        }


@dataclass
class JunctionBox:
    box_id: int
    bbox:   BoundingBox
    se:     Optional[BuildingElement] = None
    fe1:    Optional[BuildingElement] = None
    fe2:    Optional[BuildingElement] = None
    fe3:    Optional[BuildingElement] = None

    def flanking_elements(self) -> list:
        return [fe for fe in [self.fe1, self.fe2, self.fe3] if fe is not None]

    def add_flanking(self, fe: BuildingElement) -> bool:
        if self.fe1 is None:
            self.fe1 = fe; return True
        if self.fe2 is None:
            self.fe2 = fe; return True
        if self.fe3 is None:
            self.fe3 = fe; return True
        return False

    def to_dict(self) -> dict:
        fes = self.flanking_elements()
        return {
            "box_id":  self.box_id,
            "bbox":    self.bbox.to_dict(),
            "se_id":   self.se.ifc_id if self.se else None,
            "fe_ids":  [fe.ifc_id for fe in fes],
            "fe_dirs": {str(fe.ifc_id): fe.elem_direction for fe in fes},
            "fe_dd":   {str(fe.ifc_id): fe.dd_local.tolist() for fe in fes},
            "fe_dist": {str(fe.ifc_id): round(fe.distance_to_se, 4) for fe in fes},
        }


@dataclass
class JunctionResult:
    se_ifc_id:          int
    se_type:            str
    jb_id:              int
    junction_type:      str
    fe_ifc_ids:         list
    element_directions: dict
    connection_zones:   dict
    confidence:         str = "ok"
    notes:              str = ""

    def to_dict(self) -> dict:
        return {
            "se_ifc_id":          self.se_ifc_id,
            "se_type":            self.se_type,
            "jb_id":              self.jb_id,
            "junction_type":      self.junction_type,
            "fe_ifc_ids":         self.fe_ifc_ids,
            "element_directions": self.element_directions,
            "connection_zones":   self.connection_zones,
            "confidence":         self.confidence,
            "notes":              self.notes,
        }


# ===========================================================================
# HILFSFUNKTIONEN
# ===========================================================================

def vec_normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def aabb_min_distance(a: BoundingBox, b: BoundingBox) -> float:
    overlap = np.maximum(0.0,
                         np.maximum(a.min_pt, b.min_pt) -
                         np.minimum(a.max_pt, b.max_pt))
    return float(norm(overlap))


def build_world_bbox_from_local(n0, n1, u0, u1, v0, v1,
                                 n_vec, u_vec, v_vec,
                                 origin: np.ndarray) -> BoundingBox:
    R = np.column_stack([n_vec, u_vec, v_vec])
    corners_local = np.array([
        [n0, u0, v0], [n0, u0, v1], [n0, u1, v0], [n0, u1, v1],
        [n1, u0, v0], [n1, u0, v1], [n1, u1, v0], [n1, u1, v1],
    ])
    corners_world = corners_local @ R.T + origin
    return BoundingBox(corners_world.min(axis=0), corners_world.max(axis=0))


def _local_projections(elem: BuildingElement):
    """Gibt (n_min, n_max, u_min, u_max, v_min, v_max) im lokalen KOS zurück."""
    origin = elem.bbox.center()
    corners = np.array([
        elem.bbox.min_pt, elem.bbox.max_pt,
        [elem.bbox.min_pt[0], elem.bbox.max_pt[1], elem.bbox.min_pt[2]],
        [elem.bbox.max_pt[0], elem.bbox.min_pt[1], elem.bbox.max_pt[2]],
        [elem.bbox.min_pt[0], elem.bbox.min_pt[1], elem.bbox.max_pt[2]],
        [elem.bbox.max_pt[0], elem.bbox.max_pt[1], elem.bbox.min_pt[2]],
        [elem.bbox.min_pt[0], elem.bbox.max_pt[1], elem.bbox.max_pt[2]],
        [elem.bbox.max_pt[0], elem.bbox.min_pt[1], elem.bbox.min_pt[2]],
    ])
    n_p = np.dot(corners - origin, elem.n_vec)
    u_p = np.dot(corners - origin, elem.u_vec)
    v_p = np.dot(corners - origin, elem.v_vec)
    return (float(n_p.min()), float(n_p.max()),
            float(u_p.min()), float(u_p.max()),
            float(v_p.min()), float(v_p.max()))


# ===========================================================================
# SCHRITT 2 – IFC LADEN & GEOMETRIE EXTRAHIEREN
# ===========================================================================

def load_ifc(path: str) -> ifcopenshell.file:
    log.info(f"Lade IFC: {path}")
    try:
        model = ifcopenshell.open(path)
    except Exception as e:
        log.error(f"IFC konnte nicht geöffnet werden: {e}")
        sys.exit(1)
    log.info(f"Schema: {model.schema}  |  Elemente: {len(list(model))} Entitäten")
    return model


def get_local_placement_matrix(ifc_element) -> np.ndarray:
    try:
        return ifcopenshell.util.placement.get_local_placement(
            ifc_element.ObjectPlacement)
    except Exception:
        return np.eye(4)


def compute_bbox_from_geom(ifc_element, geom_settings) -> Optional[BoundingBox]:
    try:
        shape = ifcopenshell.geom.create_shape(geom_settings, ifc_element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        if len(verts) == 0:
            return None
        return BoundingBox(verts.min(axis=0), verts.max(axis=0))
    except Exception as e:
        log.debug(f"  Geometrie für #{ifc_element.id()} nicht lesbar: {e}")
        return None


def determine_element_normal(bbox: BoundingBox, mat4: np.ndarray,
                              is_slab: bool) -> tuple:
    x_ax = vec_normalize(mat4[:3, 0])
    y_ax = vec_normalize(mat4[:3, 1])
    z_ax = vec_normalize(mat4[:3, 2])
    axes = [x_ax, y_ax, z_ax]
    size = bbox.max_pt - bbox.min_pt
    projections = [abs(np.dot(size, ax)) for ax in axes]
    thin_idx = int(np.argmin(projections))
    fat_indices = [i for i in range(3) if i != thin_idx]
    n_vec = axes[thin_idx]
    if is_slab and n_vec[2] < 0:
        n_vec = -n_vec
    u_idx = fat_indices[int(np.argmax([projections[i] for i in fat_indices]))]
    u_vec = vec_normalize(axes[u_idx])
    u_vec = vec_normalize(u_vec - np.dot(u_vec, n_vec) * n_vec)
    v_vec = vec_normalize(np.cross(n_vec, u_vec))
    return n_vec, u_vec, v_vec


def get_storey_info(model: ifcopenshell.file, ifc_element) -> tuple:
    try:
        storey = ifcopenshell.util.element.get_container(ifc_element)
        if storey is None:
            return -1, "unknown"
        elev = getattr(storey, "Elevation", 0.0) or 0.0
        return int(round(elev * 10)), storey.Name or "unknown"
    except Exception:
        return -1, "unknown"


def load_elements(model: ifcopenshell.file) -> list:
    log.info("Extrahiere Bauteilgeometrie …")
    geom_settings = ifcopenshell.geom.settings()
    geom_settings.set(geom_settings.USE_WORLD_COORDS, True)
    elements = []
    seen_ids: set = set()
    skipped = 0
    for ifc_type in RELEVANT_IFC_TYPES:
        for elem in model.by_type(ifc_type):
            if elem.id() in seen_ids:
                continue
            seen_ids.add(elem.id())
            bbox = compute_bbox_from_geom(elem, geom_settings)
            if bbox is None:
                skipped += 1
                continue
            mat4 = get_local_placement_matrix(elem)
            is_slab = elem.is_a("IfcSlab")
            n_vec, u_vec, v_vec = determine_element_normal(bbox, mat4, is_slab)
            storey_id, storey_name = get_storey_info(model, elem)
            be = BuildingElement(
                ifc_id      = elem.id(),
                ifc_type    = elem.is_a(),
                global_id   = elem.GlobalId,
                name        = getattr(elem, "Name", "") or "",
                bbox        = bbox,
                n_vec       = n_vec,
                u_vec       = u_vec,
                v_vec       = v_vec,
                n_vec_world = n_vec.copy(),
                storey_id   = storey_id,
                storey_name = storey_name,
            )
            elements.append(be)
    log.info(f"  {len(elements)} Bauteile geladen, {skipped} übersprungen.")
    return elements


# ===========================================================================
# SCHRITT 3 – TRENNELEMENTE AUSWÄHLEN
# ===========================================================================

def get_adjacent_space_count(model: ifcopenshell.file, elem) -> int:
    space_ids = set()
    for rel in model.by_type("IfcRelSpaceBoundary"):
        try:
            if (rel.RelatedBuildingElement and
                    rel.RelatedBuildingElement.id() == elem.id()):
                if rel.RelatingSpace:
                    space_ids.add(rel.RelatingSpace.id())
        except AttributeError:
            continue
    return len(space_ids)


def select_separating_elements(elements: list,
                                model: ifcopenshell.file) -> list:
    log.info("Identifiziere Trennelemente …")
    elem_map = {e.ifc_id: e for e in elements}
    separating = []
    for ifc_type in RELEVANT_IFC_TYPES:
        for ifc_elem in model.by_type(ifc_type):
            if ifc_elem.id() not in elem_map:
                continue
            if get_adjacent_space_count(model, ifc_elem) >= 2:
                separating.append(elem_map[ifc_elem.id()])
    if not separating:
        log.warning("  Keine Trennelemente via SpaceBoundary → Fallback: alle Wände + Decken")
        separating = [e for e in elements
                      if "Wall" in e.ifc_type or "Slab" in e.ifc_type
                      or "Member" in e.ifc_type]
    log.info(f"  {len(separating)} Trennelemente.")
    return separating


# ===========================================================================
# SCHRITT 4 – FLANKIERENDE ELEMENTE FINDEN
# ===========================================================================

def filter_by_storey(se: BuildingElement, all_elements: list) -> list:
    return [e for e in all_elements
            if abs(e.storey_id - se.storey_id) <= 15
            and e.ifc_id != se.ifc_id]


def compute_dd_vector(se: BuildingElement,
                      fe: BuildingElement) -> tuple:
    dist = aabb_min_distance(se.bbox, fe.bbox)
    delta = fe.bbox.center() - se.bbox.center()
    local = np.array([
        np.dot(delta, se.n_vec),
        np.dot(delta, se.u_vec),
        np.dot(delta, se.v_vec),
    ])
    return dist, local


def get_semantic_flanking_ids(model: ifcopenshell.file,
                               se: BuildingElement) -> set:
    fe_ids = set()
    for rel in model.by_type("IfcRelConnectsElements"):
        try:
            rel_ids = {rel.RelatingElement.id(), rel.RelatedElement.id()}
            if se.ifc_id in rel_ids:
                fe_ids |= rel_ids - {se.ifc_id}
        except AttributeError:
            continue
    se_spaces = set()
    for rel in model.by_type("IfcRelSpaceBoundary"):
        try:
            if (rel.RelatedBuildingElement and
                    rel.RelatedBuildingElement.id() == se.ifc_id):
                if rel.RelatingSpace:
                    se_spaces.add(rel.RelatingSpace.id())
        except AttributeError:
            continue
    for rel in model.by_type("IfcRelSpaceBoundary"):
        try:
            if rel.RelatingSpace and rel.RelatingSpace.id() in se_spaces:
                if rel.RelatedBuildingElement:
                    fe_ids.add(rel.RelatedBuildingElement.id())
        except AttributeError:
            continue
    fe_ids.discard(se.ifc_id)
    return fe_ids


def find_flanking_elements(se: BuildingElement,
                            candidates: list,
                            model: ifcopenshell.file) -> list:
    semantic_ids = get_semantic_flanking_ids(model, se)
    cand_map = {e.ifc_id: e for e in candidates}
    selected = {}
    for fid in semantic_ids:
        if fid in cand_map:
            selected[fid] = cand_map[fid]
    for fe in candidates:
        if aabb_min_distance(se.bbox, fe.bbox) <= MAX_DISTANCE:
            selected[fe.ifc_id] = fe
    flanking = list(selected.values())
    for fe in flanking:
        fe.distance_to_se, fe.dd_local = compute_dd_vector(se, fe)
    flanking = [fe for fe in flanking if fe.distance_to_se <= MAX_DISTANCE]
    log.debug(f"  SE #{se.ifc_id}: {len(flanking)} flankierende Elemente")
    return flanking


# ===========================================================================
# SCHRITT 5 – ELEMENT-DIRECTIONS ZUWEISEN
# ===========================================================================

def assign_element_directions(se: BuildingElement,
                               flanking: list) -> None:
    """
    Weist jedem FE die Richtung n / m / o zu.

    Regeln:
      o  = IfcSlab oder horizontal liegende Platte (n_vec stark vertikal)
      n  = Normalenvektor parallel zu SE-Normalenvektor  (cos > ANGLE_TOL)
           ODER: Wand liegt primär in n-Richtung der SE-Decke (oben/unten)
      m  = alles andere (Querwand, 90°-Stoß)

    Sonderfall SE=Decke + FE=Wand:
      cos(se.n_vec, fe.n_vec) ≈ 0 immer (Decke Z, Wand XY)
      → Entscheidung n vs m anhand dd_local:
        abs(dd_local[0])  > abs(dd_local[2])  → Wand liegt über/unter Decke → "n"
        sonst                                 → Wand trifft Decke von der Seite → "m"
    """
    se.elem_direction = "n"
    se_is_slab = bool(abs(se.n_vec[2]) > ANGLE_TOL or se.ifc_type == "IfcSlab")

    for fe in flanking:
        is_slab_type   = fe.ifc_type in ("IfcSlab", "IfcRoof")
        n_vec_vertical = abs(fe.n_vec[2]) > ANGLE_TOL

        if is_slab_type or (n_vec_vertical and "Wall" not in fe.ifc_type):
            fe.elem_direction = "o"
            continue

        cos_n = abs(float(np.dot(se.n_vec, fe.n_vec)))

        if cos_n > ANGLE_TOL:
            fe.elem_direction = "n"
        elif se_is_slab:
            # SE = Decke (n_vec ≈ Z), FE = Wand (n_vec ≈ XY)
            # dd_local[0] = Anteil in Decken-Normalenrichtung (≈ Z, oben/unten)
            # dd_local[2] = Anteil in v-Richtung der Decke    (≈ Y, seitlich)
            abs_n_dd = abs(fe.dd_local[0])
            abs_v_dd = abs(fe.dd_local[2])
            fe.elem_direction = "n" if abs_n_dd > abs_v_dd else "m"
        else:
            fe.elem_direction = "m"


# ===========================================================================
# SCHRITT 6 – JUNCTION BOXES ERSTELLEN
# ===========================================================================

def build_junction_boxes(se: BuildingElement) -> dict:
    origin = se.bbox.center()
    n_min, n_max, u_min, u_max, v_min, v_max = _local_projections(se)
    B, T = JB_OFFSET_BORDER, JB_OFFSET_THIN
    jb_defs = {
        1: (n_min - T, n_max + T,  u_min - B, u_min + B,  v_min, v_max),
        2: (n_min - T, n_max + T,  u_min + B, u_max - B,  v_min, v_max),
        3: (n_min - T, n_max + T,  u_max - B, u_max + B,  v_min, v_max),
        4: (n_min - T, n_max + T,  u_min,     u_max,      v_min - B, v_min + B),
        5: (n_min - T, n_max + T,  u_min,     u_max,      v_min + B, v_max - B),
        6: (n_min - T, n_max + T,  u_min,     u_max,      v_max - B, v_max + B),
    }
    jbs = {}
    for jb_id, (n0, n1, u0, u1, v0, v1) in jb_defs.items():
        world_bbox = build_world_bbox_from_local(
            n0, n1, u0, u1, v0, v1,
            se.n_vec, se.u_vec, se.v_vec, origin)
        jbs[jb_id] = JunctionBox(box_id=jb_id, bbox=world_bbox, se=se)
    return jbs


# ===========================================================================
# SCHRITT 7 – FLANKIERENDE ELEMENTE DEN JUNCTION BOXES ZUWEISEN
# ===========================================================================

def determine_jb_id(se: BuildingElement, fe: BuildingElement) -> Optional[int]:
    origin = se.bbox.center()
    delta  = fe.bbox.center() - origin
    n_comp = float(np.dot(delta, se.n_vec))
    u_comp = float(np.dot(delta, se.u_vec))
    v_comp = float(np.dot(delta, se.v_vec))

    n_min, n_max, u_min, u_max, v_min, v_max = _local_projections(se)
    n_extent = max(n_max - n_min, GEOM_TOL)
    u_extent = max(u_max - u_min, GEOM_TOL)
    v_extent = max(v_max - v_min, GEOM_TOL)

    abs_n = abs(n_comp) / n_extent
    abs_u = abs(u_comp) / u_extent
    abs_v = abs(v_comp) / v_extent

    log.debug(f"    DD-Normen  n={abs_n:.3f}  u={abs_u:.3f}  v={abs_v:.3f}"
              f"  (raw n={n_comp:.3f} u={u_comp:.3f} v={v_comp:.3f})")

    if abs_u >= abs_v and abs_u >= abs_n:
        return 1 if u_comp < 0 else 3
    elif abs_v >= abs_n:
        return 4 if v_comp < 0 else 6
    else:
        # n-dominant: nächste Kante bestimmt JB
        u_fe = float(np.dot(fe.bbox.center() - origin, se.u_vec))
        v_fe = float(np.dot(fe.bbox.center() - origin, se.v_vec))
        B = JB_OFFSET_BORDER
        dists = {1: abs(u_fe - u_min), 3: abs(u_fe - u_max),
                 4: abs(v_fe - v_min), 6: abs(v_fe - v_max)}
        nearest = min(dists, key=lambda k: dists[k])
        if dists[nearest] > B:
            return 2 if u_extent >= v_extent else 5
        return nearest


def assign_flanking_to_jbs(se: BuildingElement, flanking: list,
                            jbs: dict) -> dict:
    for fe in flanking:
        jb_id = determine_jb_id(se, fe)
        if jb_id is None:
            log.warning(f"  FE #{fe.ifc_id} → keine JB ermittelt")
            continue
        if not jbs[jb_id].add_flanking(fe):
            log.warning(f"  JB{jb_id} voll – FE #{fe.ifc_id} verworfen")
        else:
            log.debug(f"  FE #{fe.ifc_id} ({fe.elem_direction}) → JB{jb_id}")
    return jbs


# ===========================================================================
# SCHRITT 8 – CONNECTION ZONES
# ===========================================================================

def get_connection_zone(host: BuildingElement, visitor: BuildingElement) -> str:
    """
    Bestimmt in welcher Zone 'visitor' auf 'host' trifft.
    Alle Berechnungen im lokalen KOS von 'host'.

    short  = Stirnfläche (Dickenrichtung, ±15 % der Dicke)
    border = Randstreifen 0.5 m
    middle = Mittelfläche
    """
    origin = host.bbox.center()
    n_min, n_max, u_min, u_max, v_min, v_max = _local_projections(host)

    contact_world = np.clip(visitor.bbox.center(),
                            host.bbox.min_pt, host.bbox.max_pt)
    delta = contact_world - origin
    c_n = float(np.dot(delta, host.n_vec))
    c_u = float(np.dot(delta, host.u_vec))
    c_v = float(np.dot(delta, host.v_vec))

    n_thick = max(n_max - n_min, GEOM_TOL)
    if abs(c_n - n_min) < n_thick * 0.15 or abs(c_n - n_max) < n_thick * 0.15:
        return "short"

    B = JB_OFFSET_BORDER
    if (c_u < u_min + B or c_u > u_max - B or
            c_v < v_min + B or c_v > v_max - B):
        return "border"

    return "middle"


# ===========================================================================
# SCHRITT 9 – JUNCTION-TYP IDENTIFIZIEREN
# ===========================================================================
#
# Klassifikationstabelle (aus Paper-Abbildung):
#
# Typ        SE     FE-Dirs (sortiert)  Schlüsselmerkmal
# ────────────────────────────────────────────────────────────────────────────
# Lh1-2      Wand   n, m               cz_fe: short, border
# Lv1-2      Wand   n, o               cz_fe: short, short
# Tv2-13     Wand   o (+opt. n/m)      cz_fe der Decke: middle
# Th1-24     Wand   n, m               cz_se der Querwand: middle
# Tv1-24     Wand   n, o               cz_fe der Wand: middle (Wand sieht SE-Mitte)
#                                       ODER: 1 FE o, cz_fe=short, JB4/6
# Th2-1-4    Wand   n, m, m            n-El: cz_fe=border/border;  m-El: short
# Xh1-24-3   Wand   m, m, n            m-El: cz_fe=middle;         n-El: short
# Tv2-1:3    Wand   o, o, n            o-El: cz_fe=border;         n-El: short/short
#                   ODER 2×o           2 Decken gegenüberl. n-Seiten, gleiche v-Seite
# Th1-2:4    Wand   n, m, n            n-El: short; m-El: cz_fe=border
# Tv2-1-4    Decke  n, m, m            n-El: cz_fe=border/border;  m-El: short
#                   ODER 2×n           2 Wände oben/unten (gegenüberl. n-Seiten)
# Tv1-2:4    Decke  o, n, n            o-El: short; n-El: cz_fe=border
# Xv1-24-3   Decke  n, o, o            n-El: cz_fe=middle;         o-El: short
# Xv2-13-4   Decke  n, o, n            o-El: cz_fe=middle
# Xh2-1:3-4  Wand   n, m, m, n(3FE)   m-El: cz_fe=border
# Xv2-1:3-4  Decke  o, o, n, n(3FE)   o-El: cz_fe=border
# ────────────────────────────────────────────────────────────────────────────

def _se_is_slab(se: BuildingElement) -> bool:
    return bool(abs(se.n_vec[2]) > ANGLE_TOL or se.ifc_type == "IfcSlab")


def _opposite_n_sides(fes: list) -> bool:
    """True wenn mindestens 2 FEs auf entgegengesetzten n-Seiten liegen."""
    signs = [float(np.sign(fe.dd_local[0])) for fe in fes]
    return any(s < 0 for s in signs) and any(s > 0 for s in signs)


def _same_v_side(fe_a: BuildingElement, fe_b: BuildingElement) -> bool:
    a, b = fe_a.dd_local[2], fe_b.dd_local[2]
    if abs(a) < GEOM_TOL and abs(b) < GEOM_TOL:
        return True
    sa, sb = float(np.sign(a)), float(np.sign(b))
    return sa != 0 and sb != 0 and sa == sb


def identify_junction_type(jb: JunctionBox) -> tuple:
    se    = jb.se
    fes   = jb.flanking_elements()
    n_fe  = len(fes)
    jb_id = jb.box_id
    slab  = _se_is_slab(se)

    if n_fe == 0:
        return "NONE", "Keine flankierenden Elemente"

    dirs   = [fe.elem_direction for fe in fes]
    cz_fe  = [get_connection_zone(se, fe) for fe in fes]   # wie trifft FE auf SE?
    cz_se  = [get_connection_zone(fe, se) for fe in fes]   # wie trifft SE auf FE?

    log.debug(f"    identify_junction_type: jb_id={jb_id} n_fe={n_fe} slab={slab} "
              f"dirs={dirs} cz_fe={cz_fe} cz_se={cz_se}")

    n_cnt = dirs.count("n")
    m_cnt = dirs.count("m")
    o_cnt = dirs.count("o")

    # ── Hilfsfunktionen ──────────────────────────────────────────────────

    def fe_idx(direction: str, exclude: set = frozenset()) -> Optional[int]:
        """Index des ersten FE mit gegebener Richtung (optional: exclude)."""
        for i, d in enumerate(dirs):
            if d == direction and i not in exclude:
                return i
        return None

    def all_idx(direction: str) -> list:
        return [i for i, d in enumerate(dirs) if d == direction]

    # ════════════════════════════════════════════════════════════════════
    # 1 FE
    # ════════════════════════════════════════════════════════════════════
    if n_fe == 1:
        d, czf, czs = dirs[0], cz_fe[0], cz_se[0]
        return _classify_1fe(se, fes[0], d, czf, czs, jb_id, slab)

    # ════════════════════════════════════════════════════════════════════
    # 2 FE
    # ════════════════════════════════════════════════════════════════════
    if n_fe == 2:
        return _classify_2fe(se, fes, dirs, cz_fe, cz_se,
                             n_cnt, m_cnt, o_cnt, jb_id, slab)

    # ════════════════════════════════════════════════════════════════════
    # 3 FE
    # ════════════════════════════════════════════════════════════════════
    if n_fe == 3:
        return _classify_3fe(se, fes, dirs, cz_fe, cz_se,
                             n_cnt, m_cnt, o_cnt, slab)

    return ("COMPLEX", f"Mehr als 3 FE ({n_fe}) – manuelle Prüfung")


# ---------------------------------------------------------------------------
# 1-FE-Klassifikation
# ---------------------------------------------------------------------------

def _classify_1fe(se, fe, d, czf, czs, jb_id, slab) -> tuple:
    """
    Tabelle 1-FE:
      Lh1-2  : Wand-SE, d=n, czf=short (keine Deckenbeteiligung)
      Lv1-2  : Wand-SE, d=o, czf=short  ODER  Decke-SE, d=n/m/o, czf=short
      Tv2-13 : d=o, czf=middle
      Th1-24 : Wand-SE, d=n, czf=short, czs=middle  (T aus Wandperspektive)
               Wand-SE, d=m, czs=middle
      Tv1-24 : Wand-SE, d=o, czf≠middle, czs≠middle → Decke trifft Wand seitlich
               Wand-SE, d=n, czf=middle
    """
    # Decke trifft SE in der Mitte
    if d == "o" and czf == "middle":
        return ("Tv2-13", "Tv2-13: Decke trifft SE-Mitte")
    if d in ("n", "m") and czs == "middle":
        # SE-Ende trifft FE-Mitte
        if slab:
            return ("Tv2-13", "Tv2-13: Wand(FE) trifft Decke(SE)-Mitte")
        return ("Th1-24", "Th1-24: SE trifft FE-Mitte")

    if not slab:
        # SE = Wand
        if d == "o":
            # Decke trifft Wand seitlich (JB4/6) oder an Rand
            if czf == "short":
                return ("Lv1-2", "Lv1-2: Decke trifft Wandende")
            if czf == "border":
                return ("Tv1-24", "Tv1-24: Decke trifft Wand(SE) im Rand")
            return ("Tv1-24", "Tv1-24: Decke–Wand")
        if d == "n":
            if czs == "border":
                if jb_id in (1, 3):
                    return ("Th1-2-4", "Th1-2-4: Parallelwand, SE-Ende im Rand")
                return ("Lh1-2", "Lh1-2: Parallelwand, Ecke")
            return ("Lh1-2", "Lh1-2: Parallelwand")
        if d == "m":
            if czs == "border":
                return ("Th1-2:4", "Th1-2:4: Querwand, SE-Ende im Rand")
            return ("Lh1-2", "Lh1-2: Querwand, Ecke")
    else:
        # SE = Decke
        if d == "o":
            return ("Lv1-2", "Lv1-2: Decke–Decke")
        if d in ("n", "m"):
            if czs == "border":
                return ("Tv1-24", "Tv1-24: Wand trifft Decke(SE) im Rand")
            return ("Lv1-2", "Lv1-2: Wand–Decke, Seite")

    return ("Lv1-2" if (slab or d == "o") else "Lh1-2", "L-Stoß (Fallback 1FE)")


# ---------------------------------------------------------------------------
# 2-FE-Klassifikation
# ---------------------------------------------------------------------------

def _classify_2fe(se, fes, dirs, cz_fe, cz_se,
                  n_cnt, m_cnt, o_cnt, jb_id, slab) -> tuple:
    """
    Klassifikationstabelle 2 FE:

    SE=Wand:
      Tv2-13   :  mind. 1×o mit czf=middle
      Th1-24   :  n+m,  czs[m]=middle  ODER  czs[n]=middle
      Tv1-24   :  n+o,  czf[n]=middle  (Wand trifft SE-Mitte aus JB2/5)
      Tv1-2:4  :  n+o,  n ist "n"-Wand (Fortsetzung), o ist Decke → Wandende
                  trifft Treffpunkt Decke+Fortsetzung
      Lv1-2    :  n+o,  czf[o]=short, czs[o]=short  (beide enden stirnseitig)
      Th2-1-4  :  n+m+m (aber hier 2FE: n+m), n hat czf=border, m hat czf=short
                  UND czs[n] = border/middle (SE endet IN der n-Wand)
      Tv2-1:3  :  2×o, gegenüberl. n-Seiten, gleiche v-Seite
      Tv2-1-4  :  2×o, gegenüberl. n-Seiten, versch. v-Seite  (Wand trennt 2 Decken)
      Lh1-2    :  n+m, Ecke
      Xh1-24-3 :  JB2/5, n+m

    SE=Decke:
      Tv2-13   :  mind. 1×(n/m) mit czs=middle
      Tv2-1-4  :  2×n, gegenüberl. n-Seiten, gleiche v-Seite
                       (beide Wände oben/unten der Decke)
      Tv1-2:4  :  n+o, o hat czf=short, n hat czf=border
      Tv1-24   :  n+o, n hat czf=middle ODER o hat czf=short+czs=short
      Lv1-2    :  n+o oder 2×o, Ecke/Ende
    """

    def idx(direction):
        for i, d in enumerate(dirs):
            if d == direction:
                return i
        return None

    # ── Jede Decke die SE-Mitte trifft → Tv2-13 ──────────────────────
    for i, d in enumerate(dirs):
        if d == "o" and cz_fe[i] == "middle":
            return ("Tv2-13", "Tv2-13: Decke trifft SE-Mitte")

    # ── JB2/5: FEs treffen SE in der MITTE ───────────────────────────
    if jb_id in (2, 5):
        if slab:
            if o_cnt == 0:
                return ("Tv2-1:3", "Tv2-1:3: 2 Wände treffen Decke(SE)-Mitte")
            return ("Xh2-1:3-4", "X-Stoß gemischt in Decken-Mitte")
        else:
            if o_cnt == 0:
                return ("Xh1-24-3", "Xh1-24-3: 2 Wände durch Wand(SE)-Mitte")
            if o_cnt == 2:
                return ("Xv2-13-4", "Xv2-13-4: 2 Decken durch Wand(SE)-Mitte")
            return ("Xh2-1:3-4", "X-Stoß gemischt in Wand-Mitte")

    # ════════════════════════════════════════════════
    # SE = Wand
    # ════════════════════════════════════════════════
    if not slab:

        # n + m  (Querwand + Parallelwand)
        if n_cnt == 1 and m_cnt == 1:
            i_n = idx("n")
            i_m = idx("m")
            # SE-Ende trifft Mitte eines FE
            if cz_se[i_n] == "middle" or cz_se[i_m] == "middle":
                return ("Th1-24", "Th1-24: SE-Ende trifft FE-Mitte")
            # Th2-1-4: n-Wand hat czf=border (SE liegt an Rand der Parallelwand)
            #          UND czs[n] zeigt, dass SE-Ende IN der n-Wand liegt
            if cz_fe[i_n] == "border" and cz_se[i_n] in ("border", "middle"):
                return ("Th2-1-4", "Th2-1-4: SE-Ende in Parallelwand, Querwand stirnseitig")
            # Th1-2:4: m-Wand hat czf=border (SE-Ende im Rand der Querwand)
            if cz_fe[i_m] == "border":
                return ("Th1-2:4", "Th1-2:4: SE-Ende im Randbereich der Querwand")
            # Sonst L-Stoß
            return ("Lh1-2", "Lh1-2: Wand–Wand Ecke")

        # 2 × n  (zwei Parallelwände)
        if n_cnt == 2:
            if _opposite_n_sides(fes):
                if _same_v_side(fes[0], fes[1]):
                    return ("Th1-2:4", "Th1-2:4: 2 Parallelwände gegenüberl., gleiche v-Seite")
                return ("Th2-1-4", "Th2-1-4: SE trennt 2 Parallelwände")
            if any(czs == "middle" for czs in cz_se):
                return ("Th1-24", "Th1-24: SE-Ende trifft Parallelwand-Mitte")
            return ("Lh1-2", "Lh1-2: 2 Parallelwände")

        # 2 × m  (zwei Querwände)
        if m_cnt == 2:
            if any(czs == "middle" for czs in cz_se):
                return ("Th1-24", "Th1-24: SE-Ende trifft Querwand-Mitte")
            return ("Lh1-2", "Lh1-2: 2 Querwände, Ecke")

        # n + o  (Parallelwand + Decke)
        if n_cnt == 1 and o_cnt == 1:
            i_n = idx("n")
            i_o = idx("o")
            # Decke oder n-Wand trifft SE-Mitte
            if cz_fe[i_n] == "middle":
                return ("Tv1-24", "Tv1-24: Parallelwand trifft Wandmitte")
            # Tv1-2:4: n-Wand ist Fortsetzung der SE-Wand, Decke kommt dazu
            # Signatur: czs[n] = border oder middle (SE endet IN der n-Wand)
            if cz_se[i_n] in ("border", "middle"):
                return ("Tv1-2:4", "Tv1-2:4: Parallelwand(n) + Decke, SE-Ende in Fortsetzungswand")
            # Lv1-2: beide treffen stirnseitig
            if cz_fe[i_o] == "short" and cz_se[i_o] == "short":
                return ("Lv1-2", "Lv1-2: Decke + Wand, stirnseitig")
            return ("Tv1-24", "Tv1-24: Parallelwand + Decke, allgemein")

        # m + o  (Querwand + Decke)
        if m_cnt == 1 and o_cnt == 1:
            i_m = idx("m")
            i_o = idx("o")
            if cz_fe[i_o] == "border":
                return ("Tv1-2:4", "Tv1-2:4: Querwand(m) + Decke im Rand")
            if cz_se[i_m] == "middle":
                return ("Th1-24", "Th1-24: SE-Ende trifft Querwand-Mitte")
            return ("Tv1-24", "Tv1-24: Querwand + Decke")

        # 2 × o  (zwei Decken)
        if o_cnt == 2:
            if _opposite_n_sides(fes):
                if _same_v_side(fes[0], fes[1]):
                    return ("Tv2-1:3",
                            "Tv2-1:3: 2 Decken, gegenüberl. n-Seiten, gleiche v-Seite")
                return ("Tv2-1-4",
                        "Tv2-1-4: Wand(SE) trennt 2 Decken oben+unten")
            return ("Tv1-24", "Tv1-24: 2 Decken gleiche Seite")

    # ════════════════════════════════════════════════
    # SE = Decke
    # ════════════════════════════════════════════════
    else:

        # SE-Ende trifft Mitte eines FE (Wand trifft Deckenmitte → T-Stoß)
        for i, czs in enumerate(cz_se):
            if czs == "middle":
                return ("Tv2-13", "Tv2-13: Wand(FE) trifft Decke(SE)-Mitte")

        # 2 × n  (zwei Wandscheiben oben/unten der Decke)
        if n_cnt == 2:
            if _opposite_n_sides(fes):
                if _same_v_side(fes[0], fes[1]):
                    return ("Tv2-1-4",
                            "Tv2-1-4: Decke(SE) liegt zwischen zwei Wandscheiben")
                return ("Tv2-1:3",
                        "Tv2-1:3: 2 Wandscheiben, versch. v-Seite")
            return ("Tv1-24", "Tv1-24: 2 Wandscheiben gleiche Seite")

        # 2 × m  (zwei Querwände)
        if m_cnt == 2:
            if _opposite_n_sides(fes) and _same_v_side(fes[0], fes[1]):
                return ("Tv1-2:4", "Tv1-2:4: 2 Querwände, gegenüberl. Seiten")
            return ("Tv1-24", "Tv1-24: 2 Querwände")

        # n + m
        if n_cnt == 1 and m_cnt == 1:
            i_n = idx("n")
            i_m = idx("m")
            if cz_se[i_n] in ("border", "middle"):
                return ("Tv2-1-4",
                        "Tv2-1-4: n-Wand oben/unten, m-Wand stirnseitig")
            if cz_fe[i_n] == "border":
                return ("Tv1-2:4", "Tv1-2:4: n-Wand Rand")
            return ("Tv1-24", "Tv1-24: n+m allgemein")

        # o + n  (andere Decke + Wand)
        if o_cnt == 1 and n_cnt == 1:
            i_o = idx("o")
            i_n = idx("n")
            if cz_fe[i_n] == "border":
                return ("Tv1-2:4", "Tv1-2:4: Decke + Wand(n) im Rand")
            if cz_fe[i_n] == "middle":
                return ("Xv1-24-3", "Xv1-24-3: Wand trifft Deckenmitte")
            return ("Tv1-24", "Tv1-24: Decke + Wand")

        # o + m
        if o_cnt == 1 and m_cnt == 1:
            i_o = idx("o")
            i_m = idx("m")
            if cz_fe[i_m] == "middle":
                return ("Xv1-24-3", "Xv1-24-3: Querwand trifft Deckenmitte")
            return ("Lv1-2", "Lv1-2: Decke + Querwand, Seite")

        # 2 × o
        if o_cnt == 2:
            return ("Lv1-2", "Lv1-2: 2 Decken")

    return ("Tv1-24" if slab else "Th1-24", "T-Stoß (Fallback 2FE)")


# ---------------------------------------------------------------------------
# 3-FE-Klassifikation
# ---------------------------------------------------------------------------

def _classify_3fe(se, fes, dirs, cz_fe, cz_se,
                  n_cnt, m_cnt, o_cnt, slab) -> tuple:
    """
    Klassifikationstabelle 3 FE:

    SE=Wand:
      Th2-1-4   : 1×n + 2×m,  n hat czf=border, m haben czf=short
      Xh1-24-3  : 2×m + 1×n,  m haben czs=middle, n hat czf=short
      Tv2-1:3   : 2×o + 1×n,  o haben czf=border, n hat czf=short/short
      Th1-2:4   : 2×n + 1×m,  n haben czf=short, m hat czf=border

    SE=Decke:
      Tv2-1-4   : 1×n + 2×m,  n hat czf=border, m haben czf=short
      Tv1-2:4   : 2×n + 1×o,  o hat czf=short, n haben czf=border
      Xv1-24-3  : 1×n + 2×o,  n hat czf=middle, o haben czf=short
      Xv2-13-4  : 2×n + 1×o,  o hat czf=middle
    """

    def idx(direction, exclude=frozenset()):
        for i, d in enumerate(dirs):
            if d == direction and i not in exclude:
                return i
        return None

    def all_idx(direction):
        return [i for i, d in enumerate(dirs) if d == direction]

    # ── mind. 1 Decke mit czf=middle → immer Tv2-13 / Xv2-13-4 ──────
    for i, d in enumerate(dirs):
        if d == "o" and cz_fe[i] == "middle":
            return ("Xv2-13-4" if slab else "Tv2-13",
                    "Decke(FE) trifft SE-Mitte")

    # ── mind. 1 n/m-Wand mit czs=middle → T/X Mitte ─────────────────
    for i, d in enumerate(dirs):
        if d in ("n", "m") and cz_se[i] == "middle":
            if slab:
                return ("Xv2-13-4", "Xv2-13-4: Wand trifft Deckenmitte")
            return ("Xh1-24-3", "Xh1-24-3: Wand trifft Wandmitte")

    # ════════════════════════════════════════════════
    # SE = Wand
    # ════════════════════════════════════════════════
    if not slab:

        # Th2-1-4: 1×n + 2×m, n-Wand im Rand (SE endet IN der Parallelwand)
        if n_cnt == 1 and m_cnt == 2:
            i_n = idx("n")
            i_ms = all_idx("m")
            n_in_parallel = cz_se[i_n] in ("border", "middle") or cz_fe[i_n] == "border"
            m_short = all(cz_fe[i] == "short" for i in i_ms)
            if n_in_parallel and m_short:
                return ("Th2-1-4", "Th2-1-4: Parallelwand(n) im Rand + 2 Querwände")
            return ("Th2-1-4", "Th2-1-4: 1×n + 2×m allgemein")

        # Xh1-24-3: 2×m + 1×n, Querwände treffen SE-Mitte (czs=middle)
        if m_cnt == 2 and n_cnt == 1:
            return ("Xh1-24-3", "Xh1-24-3: 2 Querwände + Parallelwand durch Wandmitte")

        # Tv2-1:3: 2×o + 1×n
        if o_cnt == 2 and n_cnt == 1:
            i_os = all_idx("o")
            # Beide Decken auf gegenüberl. n-Seiten, gleiche v-Seite
            fe_os = [fes[i] for i in i_os]
            if _opposite_n_sides(fe_os) and _same_v_side(fe_os[0], fe_os[1]):
                return ("Tv2-1:3", "Tv2-1:3: 2 Decken gegenüberl. + Parallelwand")
            return ("Tv2-1:3", "Tv2-1:3: 2 Decken + Parallelwand")

        # Th1-2:4: 2×n + 1×m
        if n_cnt == 2 and m_cnt == 1:
            return ("Th1-2:4", "Th1-2:4: 2 Parallelwände + 1 Querwand")

        # o + n + m gemischt
        if o_cnt >= 1:
            return ("Xh2-1:3-4", "Xh2-1:3-4: gemischt Wand+Decken")

    # ════════════════════════════════════════════════
    # SE = Decke
    # ════════════════════════════════════════════════
    else:

        # Tv2-1-4: 1×n + 2×m (Wandscheibe oben/unten + 2 Querwände)
        if n_cnt == 1 and m_cnt == 2:
            i_n = idx("n")
            i_ms = all_idx("m")
            n_in_parallel = cz_se[i_n] in ("border", "middle") or cz_fe[i_n] == "border"
            m_short = all(cz_fe[i] == "short" for i in i_ms)
            if n_in_parallel and m_short:
                return ("Tv2-1-4", "Tv2-1-4: Wandscheibe(n) im Rand + 2 Querwände")
            return ("Tv2-1-4", "Tv2-1-4: 1×n + 2×m")

        # Tv1-2:4: 2×n + 1×o  (2 Wandscheiben oben/unten, 1 Decke stirnseitig)
        if n_cnt == 2 and o_cnt == 1:
            i_o = idx("o")
            i_ns = all_idx("n")
            o_short = cz_fe[i_o] == "short"
            n_border = all(cz_fe[i] == "border" for i in i_ns)
            if o_short and n_border:
                return ("Tv1-2:4", "Tv1-2:4: Decke stirnseitig + 2 Wandscheiben im Rand")
            return ("Tv1-2:4", "Tv1-2:4: 2×n + 1×o allgemein")

        # Xv1-24-3: 1×n + 2×o  (Wand in Deckenmitte, 2 Decken seitlich)
        if n_cnt == 1 and o_cnt == 2:
            i_n = idx("n")
            if cz_fe[i_n] == "middle":
                return ("Xv1-24-3", "Xv1-24-3: Wand trifft Deckenmitte + 2 Decken")
            return ("Xv1-24-3", "Xv1-24-3: 1×n + 2×o")

        # Xv2-13-4: 2×n + 1×o  (Decke trifft SE-Mitte)  – bereits oben abgefangen
        if n_cnt == 2 and o_cnt == 1:
            return ("Xv2-13-4", "Xv2-13-4: 2 Wände + Decke in Deckenmitte")

        # Xv2-1:3-4: 2×o + 1×n + 1×n = 2×o + 2×n (aber hier 3FE → 2×o + 1×n)
        if o_cnt == 2 and n_cnt == 1:
            return ("Xv2-1:3-4", "Xv2-1:3-4: 2 Decken + Wand, gemischt")

    return ("COMPLEX", f"Keine Regel für dirs={dirs}, slab={slab}")


# ===========================================================================
# SCHRITT 10 – HAUPTPIPELINE
# ===========================================================================

def run(ifc_path: str, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_path = out / "junction_analysis.log"
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    log.info("=" * 60)
    log.info(f"IFC Junction Analysis  –  {ts}")
    log.info("=" * 60)

    debug: dict = {
        "meta": {
            "ifc_path":  str(ifc_path),
            "timestamp": ts,
            "constants": {
                "MAX_DISTANCE":     MAX_DISTANCE,
                "JB_OFFSET_THIN":   JB_OFFSET_THIN,
                "JB_OFFSET_BORDER": JB_OFFSET_BORDER,
            }
        },
        "elements":       [],
        "sep_elements":   [],
        "junction_boxes": [],
        "errors":         [],
    }
    results: dict = {
        "meta":    {"ifc_path": str(ifc_path), "timestamp": ts},
        "summary": {},
        "junctions": [],
    }

    model    = load_ifc(ifc_path)
    elements = load_elements(model)
    debug["elements"] = [e.to_dict() for e in elements]

    sep_elems = select_separating_elements(elements, model)
    debug["sep_elements"] = [e.ifc_id for e in sep_elems]

    all_junctions: list = []

    for se in sep_elems:
        log.info(f"─── SE #{se.ifc_id} ({se.ifc_type}, {se.name}) ───")
        candidates = filter_by_storey(se, elements)
        flanking   = find_flanking_elements(se, candidates, model)
        if not flanking:
            log.info("  Keine flankierenden Elemente → übersprungen")
            continue

        assign_element_directions(se, flanking)
        jbs = build_junction_boxes(se)
        jbs = assign_flanking_to_jbs(se, flanking, jbs)

        for jb in jbs.values():
            entry = jb.to_dict()
            entry["se_name"] = se.name
            debug["junction_boxes"].append(entry)

        for jb_id, jb in jbs.items():
            fes = jb.flanking_elements()
            if not fes:
                continue
            jtype, notes = identify_junction_type(jb)
            cz = {str(fe.ifc_id): get_connection_zone(se, fe) for fe in fes}
            ed = {str(fe.ifc_id): fe.elem_direction for fe in fes}
            confidence = "warn" if jtype in ("COMPLEX", "UNKNOWN") else "ok"

            jr = JunctionResult(
                se_ifc_id          = se.ifc_id,
                se_type            = se.ifc_type,
                jb_id              = jb_id,
                junction_type      = jtype,
                fe_ifc_ids         = [fe.ifc_id for fe in fes],
                element_directions = ed,
                connection_zones   = cz,
                confidence         = confidence,
                notes              = notes,
            )
            all_junctions.append(jr)
            log.info(f"  JB{jb_id}: {jtype:15s}  FEs: {[fe.ifc_id for fe in fes]}  [{confidence}]")
            if notes:
                log.debug(f"         Note: {notes}")

    # ── Deduplizierung ──────────────────────────────────────────────────
    JUNCTION_PRIO = {
        "Th1-24": 0, "Th1-2:4": 0, "Th1-2-4": 0, "Th2-1-4": 0,
        "Tv1-24": 0, "Tv1-2:4": 0, "Tv1-2-4": 0, "Tv2-1-4": 0,
        "Tv2-13": 0, "Tv2-1:3": 0,
        "Xv1-24-3": 0, "Xv2-13-4": 0, "Xh1-24-3": 0,
        "Xh2-1:3-4": 0, "Xv2-1:3-4": 0,
        "Lh1-2": 1, "Lv1-2": 1,
        "NONE": 9, "COMPLEX": 8, "UNKNOWN": 8,
    }
    CONF_PRIO = {"ok": 0, "warn": 1, "error": 2}

    def score(r: JunctionResult):
        return (CONF_PRIO.get(r.confidence, 9),
                JUNCTION_PRIO.get(r.junction_type, 5),
                0 if r.jb_id in (2, 5) else 1,
                -len(r.fe_ifc_ids))

    seen_keys: dict = {}
    for jr in all_junctions:
        key = frozenset({jr.se_ifc_id} | set(jr.fe_ifc_ids))
        if key not in seen_keys or score(jr) < score(seen_keys[key]):
            seen_keys[key] = jr

    all_keys = list(seen_keys.keys())
    subset_keys = set()
    for i, ka in enumerate(all_keys):
        for j, kb in enumerate(all_keys):
            if i != j and ka < kb:
                log.info(f"  Teilmenge entfernt: {set(ka)} ⊂ {set(kb)}")
                subset_keys.add(ka)

    deduped = [jr for key, jr in seen_keys.items() if key not in subset_keys]
    removed = len(all_junctions) - len(deduped)
    if removed:
        log.info(f"  Deduplizierung: {removed} Duplikat(e)/Teilmenge(n) entfernt "
                 f"→ {len(deduped)} eindeutige Stoßstellen")
    all_junctions = deduped

    type_counts: dict = {}
    for jr in all_junctions:
        type_counts[jr.junction_type] = type_counts.get(jr.junction_type, 0) + 1

    results["summary"] = {
        "total_junctions":      len(all_junctions),
        "separating_elements":  len(sep_elems),
        "junction_type_counts": type_counts,
    }
    results["junctions"] = [jr.to_dict() for jr in all_junctions]

    def np_encoder(obj):
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        raise TypeError(f"Nicht serialisierbar: {type(obj)}")

    debug_path  = out / "debug_junction_analysis.json"
    result_path = out / "junctions_result.json"

    with open(debug_path,  "w", encoding="utf-8") as f:
        json.dump(debug,   f, indent=2, ensure_ascii=False, default=np_encoder)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=np_encoder)

    log.info("=" * 60)
    log.info(f"✓ {len(all_junctions)} Stoßstellen gefunden")
    log.info(f"✓ Typen: {type_counts}")
    log.info(f"✓ Debug:    {debug_path}")
    log.info(f"✓ Ergebnis: {result_path}")
    log.info(f"✓ Log:      {log_path}")
    log.info("=" * 60)


# ===========================================================================
# EINSTIEGSPUNKT
# ===========================================================================

def main():
    IFC_PATH = "./ifc-models/Tv2-13.ifc"
    OUT_DIR  = "./output"

    parser = argparse.ArgumentParser(
        description="Rotationsrobuste IFC-Stoßstellenanalyse")
    parser.add_argument("--ifc", default=IFC_PATH)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    ifc_path = pathlib.Path(args.ifc)
    if not ifc_path.exists():
        print(f"[FEHLER] IFC-Datei nicht gefunden: {ifc_path.resolve()}")
        sys.exit(1)

    run(str(ifc_path), args.out)


if __name__ == "__main__":
    main()
