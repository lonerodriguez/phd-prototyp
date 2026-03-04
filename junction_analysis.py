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

# ---------------------------------------------------------------------------
# Prüfe Abhängigkeiten
# ---------------------------------------------------------------------------
try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.placement
    import ifcopenshell.util.element
except ImportError:
    print("[FEHLER] ifcopenshell nicht installiert.")
    print("  -> pip install ifcopenshell")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Globale Konstanten
# ---------------------------------------------------------------------------
MAX_DISTANCE     = 0.30   # m  – maximaler Abstand FE↔SE
JB_OFFSET_THIN   = 0.30   # m  – Überstand der JB in Dickenrichtung (n_vec)
JB_OFFSET_BORDER = 0.50   # m  – Randbereich (u_vec / v_vec)
ANGLE_TOL        = 0.85   # cos(≈32°) – Schwelle für "parallel"
GEOM_TOL         = 1e-3   # m  – geometrische Toleranz

RELEVANT_IFC_TYPES = [
    "IfcWall", "IfcWallStandardCase",
    "IfcSlab",
    "IfcCurtainWall",
    "IfcMember",        # Holzbauteile im Holzbau
    "IfcPlate",
]

log = logging.getLogger("junction_analysis")


# ===========================================================================
# DATENSTRUKTUREN
# ===========================================================================

@dataclass
class BoundingBox:
    """Achsenparallele Bounding Box in Weltkoordinaten."""
    min_pt: np.ndarray   # shape (3,)
    max_pt: np.ndarray   # shape (3,)

    def center(self) -> np.ndarray:
        return (self.min_pt + self.max_pt) * 0.5

    def size(self) -> np.ndarray:
        return self.max_pt - self.min_pt

    def contains_point(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.min_pt - GEOM_TOL) and
                    np.all(p <= self.max_pt + GEOM_TOL))

    def intersects(self, other: "BoundingBox") -> bool:
        return bool(np.all(self.min_pt <= other.max_pt + GEOM_TOL) and
                    np.all(self.max_pt >= other.min_pt - GEOM_TOL))

    def to_dict(self) -> dict:
        return {"min": self.min_pt.tolist(), "max": self.max_pt.tolist()}


@dataclass
class BuildingElement:
    """Repräsentiert ein IFC-Bauteil mit Geometrie und lokalem KOS."""
    ifc_id:       int
    ifc_type:     str
    global_id:    str
    name:         str
    bbox:         BoundingBox
    # Lokales Koordinatensystem (normierte Einheitsvektoren)
    n_vec:        np.ndarray   # Normalen-/Dickenrichtung
    u_vec:        np.ndarray   # Hauptrichtung (längste Ausdehnung in Ebene)
    v_vec:        np.ndarray   # Dritte Achse = n_vec × u_vec
    storey_id:    int = -1
    storey_name:  str = ""
    # Wird in der Analyse befüllt
    elem_direction: str = ""   # "n", "m", "o"
    distance_to_se: float = np.inf
    dd_local:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    # n_vec im Weltkoordinatensystem für Debug
    n_vec_world:  np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_dict(self) -> dict:
        return {
            "ifc_id":     self.ifc_id,
            "ifc_type":   self.ifc_type,
            "global_id":  self.global_id,
            "name":       self.name,
            "storey":     self.storey_name,
            "bbox":       self.bbox.to_dict(),
            "n_vec":      self.n_vec.tolist(),
            "u_vec":      self.u_vec.tolist(),
            "v_vec":      self.v_vec.tolist(),
            "elem_dir":   self.elem_direction,
        }


@dataclass
class JunctionBox:
    """Container für bis zu 4 Bauteile an einem Ende des Trennelements."""
    box_id:  int           # 1 .. 6
    bbox:    BoundingBox
    se:      Optional[BuildingElement] = None
    fe1:     Optional[BuildingElement] = None
    fe2:     Optional[BuildingElement] = None
    fe3:     Optional[BuildingElement] = None

    def flanking_elements(self) -> list:
        return [fe for fe in [self.fe1, self.fe2, self.fe3] if fe is not None]

    def add_flanking(self, fe: BuildingElement) -> bool:
        """Fügt FE in freien Slot ein. Gibt False zurück wenn voll."""
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
    """Ergebnis der Stoßstellenanalyse für eine JunctionBox."""
    se_ifc_id:         int
    se_type:           str
    jb_id:             int
    junction_type:     str
    fe_ifc_ids:        list
    element_directions: dict
    connection_zones:  dict
    confidence:        str = "ok"  # "ok" | "warn" | "error"
    notes:             str = ""

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
# HILFSFUNKTIONEN – VEKTORRECHNUNG
# ===========================================================================

def vec_normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def aabb_min_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Minimaler Abstand zweier AABBs (0.0 bei Überlappung/Berührung)."""
    overlap = np.maximum(0.0,
                         np.maximum(a.min_pt, b.min_pt) -
                         np.minimum(a.max_pt, b.max_pt))
    return float(norm(overlap))


def build_world_bbox_from_local(n0, n1, u0, u1, v0, v1,
                                 n_vec, u_vec, v_vec,
                                 origin: np.ndarray) -> BoundingBox:
    """
    Baut eine Welt-AABB aus 6 lokalen Grenzen auf.
    Korrekt für JEDE Rotation, da alle 8 Ecken transformiert werden.
    """
    R = np.column_stack([n_vec, u_vec, v_vec])   # 3×3 Rotationsmatrix
    corners_local = np.array([
        [n0, u0, v0], [n0, u0, v1], [n0, u1, v0], [n0, u1, v1],
        [n1, u0, v0], [n1, u0, v1], [n1, u1, v0], [n1, u1, v1],
    ])
    # Rotation: lokal → Welt, dann Translation
    corners_world = corners_local @ R.T + origin
    return BoundingBox(corners_world.min(axis=0), corners_world.max(axis=0))


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
    """Gibt die 4×4 Weltkoordinaten-Transformationsmatrix zurück."""
    try:
        mat = ifcopenshell.util.placement.get_local_placement(
            ifc_element.ObjectPlacement)
        return mat
    except Exception:
        return np.eye(4)


def compute_bbox_from_geom(ifc_element, geom_settings) -> Optional[BoundingBox]:
    """Berechnet die AABB in Weltkoordinaten via Mesh-Extraktion."""
    try:
        shape = ifcopenshell.geom.create_shape(geom_settings, ifc_element)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        if len(verts) == 0:
            return None
        return BoundingBox(verts.min(axis=0), verts.max(axis=0))
    except Exception as e:
        log.debug(f"  Geometrie für #{ifc_element.id()} nicht lesbar: {e}")
        return None


def determine_element_normal(bbox: BoundingBox,
                              mat4: np.ndarray,
                              is_slab: bool) -> tuple:
    """
    Bestimmt n_vec, u_vec, v_vec aus der Platzierungsmatrix.
    n_vec = Achse mit kleinster Bauteilausdehnung (= Dickenrichtung).
    Rotationsunabhängig.
    """
    # Lokale Achsen aus Transformationsmatrix
    x_ax = vec_normalize(mat4[:3, 0])
    y_ax = vec_normalize(mat4[:3, 1])
    z_ax = vec_normalize(mat4[:3, 2])
    axes = [x_ax, y_ax, z_ax]

    size = bbox.max_pt - bbox.min_pt

    # Projiziere Ausdehnung auf lokale Achsen
    projections = [abs(np.dot(size, ax)) for ax in axes]
    thin_idx    = int(np.argmin(projections))   # dünnste Richtung = Normale
    fat_indices = [i for i in range(3) if i != thin_idx]

    n_vec = axes[thin_idx]

    # Konvention: Decken zeigen nach oben
    if is_slab and n_vec[2] < 0:
        n_vec = -n_vec

    # u_vec = längste tangentiale Richtung
    u_idx = fat_indices[int(np.argmax([projections[i] for i in fat_indices]))]
    u_vec = vec_normalize(axes[u_idx])
    # orthogonalisieren (Gram-Schmidt)
    u_vec = vec_normalize(u_vec - np.dot(u_vec, n_vec) * n_vec)
    v_vec = vec_normalize(np.cross(n_vec, u_vec))

    return n_vec, u_vec, v_vec


def get_storey_info(model: ifcopenshell.file,
                    ifc_element) -> tuple[int, str]:
    """Gibt (storey_index, storey_name) zurück."""
    try:
        storey = ifcopenshell.util.element.get_container(ifc_element)
        if storey is None:
            return -1, "unknown"
        # Storey-Elevation als Sortierkriterium
        elev = getattr(storey, "Elevation", 0.0) or 0.0
        return int(round(elev * 10)), storey.Name or "unknown"
    except Exception:
        return -1, "unknown"


def load_elements(model: ifcopenshell.file) -> list[BuildingElement]:
    """Lädt alle relevanten Bauteile und extrahiert ihre Geometrie."""
    log.info("Extrahiere Bauteilgeometrie …")
    geom_settings = ifcopenshell.geom.settings()
    geom_settings.set(geom_settings.USE_WORLD_COORDS, True)

    elements: list[BuildingElement] = []
    seen_ids: set[int] = set()   # Verhindert Duplikate (z.B. IfcWall + IfcWallStandardCase)
    skipped = 0

    for ifc_type in RELEVANT_IFC_TYPES:
        for elem in model.by_type(ifc_type):
            if elem.id() in seen_ids:
                continue                  # Bereits durch anderen Typ geladen
            seen_ids.add(elem.id())
            bbox = compute_bbox_from_geom(elem, geom_settings)
            if bbox is None:
                skipped += 1
                continue

            size = bbox.max_pt - bbox.min_pt
            if np.any(size < GEOM_TOL):
                # Entartetes Bauteil (flach in einer Richtung auf 0)
                # Trotzdem versuchen weiter zu machen
                log.debug(f"  #{elem.id()} sehr kleines Bauteil: {size}")

            mat4    = get_local_placement_matrix(elem)
            is_slab = elem.is_a("IfcSlab")
            n_vec, u_vec, v_vec = determine_element_normal(bbox, mat4, is_slab)
            storey_id, storey_name = get_storey_info(model, elem)

            be = BuildingElement(
                ifc_id       = elem.id(),
                ifc_type     = elem.is_a(),
                global_id    = elem.GlobalId,
                name         = getattr(elem, "Name", "") or "",
                bbox         = bbox,
                n_vec        = n_vec,
                u_vec        = u_vec,
                v_vec        = v_vec,
                n_vec_world  = n_vec.copy(),
                storey_id    = storey_id,
                storey_name  = storey_name,
            )
            elements.append(be)

    log.info(f"  {len(elements)} Bauteile geladen, {skipped} übersprungen.")
    return elements


# ===========================================================================
# SCHRITT 3 – TRENNELEMENTE AUSWÄHLEN
# ===========================================================================

def get_adjacent_space_count(model: ifcopenshell.file, elem) -> int:
    """Zählt angrenzende IfcSpaces via IfcRelSpaceBoundary."""
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


def select_separating_elements(elements: list[BuildingElement],
                                model: ifcopenshell.file) -> list[BuildingElement]:
    """
    Wählt Trennelemente aus.
    Primär: Elemente mit ≥2 IfcRelSpaceBoundary-Räumen.
    Fallback: alle Wände und Decken.
    """
    log.info("Identifiziere Trennelemente …")
    elem_map = {e.ifc_id: e for e in elements}
    separating = []

    for ifc_type in RELEVANT_IFC_TYPES:
        for ifc_elem in model.by_type(ifc_type):
            if ifc_elem.id() not in elem_map:
                continue
            cnt = get_adjacent_space_count(model, ifc_elem)
            if cnt >= 2:
                separating.append(elem_map[ifc_elem.id()])

    if not separating:
        log.warning("  Keine Trennelemente via SpaceBoundary gefunden → "
                    "Fallback: alle Wände + Decken")
        separating = [e for e in elements
                      if "Wall" in e.ifc_type or "Slab" in e.ifc_type
                      or "Member" in e.ifc_type]

    log.info(f"  {len(separating)} Trennelemente.")
    return separating


# ===========================================================================
# SCHRITT 4 – FLANKIERENDE ELEMENTE FINDEN
# ===========================================================================

def filter_by_storey(se: BuildingElement,
                     all_elements: list[BuildingElement]) -> list[BuildingElement]:
    """Gibt Elemente aus gleichem und benachbarten Geschoss zurück."""
    return [e for e in all_elements
            if abs(e.storey_id - se.storey_id) <= 15   # 15 = 1 Geschoss bei Raster 10
            and e.ifc_id != se.ifc_id]


def compute_dd_vector(se: BuildingElement,
                      fe: BuildingElement) -> tuple[float, np.ndarray]:
    """
    Berechnet:
    - minimalen AABB-Abstand zwischen SE und FE
    - DD-Vektor im lokalen KOS des SE: [n-Anteil, u-Anteil, v-Anteil]
    Der DD-Vektor ist rotationsunabhängig.
    """
    dist = aabb_min_distance(se.bbox, fe.bbox)
    delta = fe.bbox.center() - se.bbox.center()
    local = np.array([
        np.dot(delta, se.n_vec),
        np.dot(delta, se.u_vec),
        np.dot(delta, se.v_vec),
    ])
    return dist, local


def get_semantic_flanking_ids(model: ifcopenshell.file,
                               se: BuildingElement) -> set[int]:
    """Gibt IFC-IDs von FEs aus semantischen Relationen zurück."""
    fe_ids = set()
    # IfcRelConnectsElements
    for rel in model.by_type("IfcRelConnectsElements"):
        try:
            rel_ids = {rel.RelatingElement.id(), rel.RelatedElement.id()}
            if se.ifc_id in rel_ids:
                fe_ids |= rel_ids - {se.ifc_id}
        except AttributeError:
            continue
    # IfcRelSpaceBoundary (gleiche Räume)
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
                            candidates: list[BuildingElement],
                            model: ifcopenshell.file) -> list[BuildingElement]:
    """
    Kombiniert semantische und geometrische Suche nach flankierenden Elementen.
    Setzt distance_to_se und dd_local auf jedem FE.
    """
    semantic_ids = get_semantic_flanking_ids(model, se)
    cand_map     = {e.ifc_id: e for e in candidates}
    selected     = {}

    # Quelle 1+2: Semantik
    for fid in semantic_ids:
        if fid in cand_map:
            selected[fid] = cand_map[fid]

    # Quelle 3: Geometrischer Abstand
    for fe in candidates:
        d = aabb_min_distance(se.bbox, fe.bbox)
        if d <= MAX_DISTANCE:
            selected[fe.ifc_id] = fe

    flanking = list(selected.values())

    # DD-Vektor und Abstand berechnen
    for fe in flanking:
        fe.distance_to_se, fe.dd_local = compute_dd_vector(se, fe)

    # Nochmals filtern: nur wirklich nahe Elemente
    flanking = [fe for fe in flanking if fe.distance_to_se <= MAX_DISTANCE]

    log.debug(f"  SE #{se.ifc_id}: {len(flanking)} flankierende Elemente")
    return flanking


# ===========================================================================
# SCHRITT 5 – ELEMENT-DIRECTIONS ZUWEISEN
# ===========================================================================

def assign_element_directions(se: BuildingElement,
                               flanking: list[BuildingElement]) -> None:
    """
    Weist n/m/o zu.

    Für SE=Wand:
      - FE=Slab/Roof                        → 'o'
      - dot(se.n_vec, fe.n_vec) hoher cos   → 'n'  (parallele Wand)
      - sonst                               → 'm'  (Querwand)

    Für SE=Decke (slab):
      - FE=Slab/Roof                        → 'o'
      - fe.u_vec parallel zu se.u_vec       → 'n'  (Wand läuft in Decken-u-Richtung)
      - fe.u_vec parallel zu se.v_vec       → 'm'  (Wand läuft quer zur Decke)

    Wände (IfcWall*) sind NIEMALS 'o'.
    """
    se_is_slab = (se.ifc_type in ("IfcSlab", "IfcRoof")) or (abs(se.n_vec[2]) > ANGLE_TOL)
    se.elem_direction = "o" if se_is_slab else "n"

    for fe in flanking:
        fe_is_slab = fe.ifc_type in ("IfcSlab", "IfcRoof")
        n_vec_vertical = abs(fe.n_vec[2]) > ANGLE_TOL

        if fe_is_slab or (n_vec_vertical and "Wall" not in fe.ifc_type):
            fe.elem_direction = "o"
            continue

        if se_is_slab:
            # Vergleiche Wand-Längsachse mit Decken-Ebene
            a_u = abs(float(np.dot(fe.u_vec, se.u_vec)))
            a_v = abs(float(np.dot(fe.u_vec, se.v_vec)))
            fe.elem_direction = "n" if a_u >= a_v else "m"
        else:
            cos_n = abs(float(np.dot(se.n_vec, fe.n_vec)))
            fe.elem_direction = "n" if cos_n > ANGLE_TOL else "m"


# ===========================================================================
# SCHRITT 6 – JUNCTION BOXES ERSTELLEN (ROTATIONSROBUST)
# ===========================================================================

def build_junction_boxes(se: BuildingElement) -> dict[int, JunctionBox]:
    """
    Erstellt 6 Junction Boxes für ein Trennelement.
    Alle Offsets werden im lokalen KOS des SE definiert,
    dann über Rotation in Weltkoordinaten transformiert.

    Lokales KOS:
        n_vec = Dickenrichtung
        u_vec = Längsrichtung (z.B. entlang der Wand)
        v_vec = Dritte Achse (z.B. Höhenrichtung)

    Box-Schema:
        JB1: u-Minus-Ende   JB3: u-Plus-Ende    JB2: u-Mitte
        JB4: v-Minus-Ende   JB6: v-Plus-Ende    JB5: v-Mitte
    """
    bbox = se.bbox
    origin = se.bbox.center()   # Weltkoordinaten-Ursprung für die Transformation

    # Projektionen der AABB auf lokale Achsen
    # (korrekte Umrechnung für rotierte Bauteile)
    corners = np.array([
        bbox.min_pt, bbox.max_pt,
        [bbox.min_pt[0], bbox.max_pt[1], bbox.min_pt[2]],
        [bbox.max_pt[0], bbox.min_pt[1], bbox.max_pt[2]],
        [bbox.min_pt[0], bbox.min_pt[1], bbox.max_pt[2]],
        [bbox.max_pt[0], bbox.max_pt[1], bbox.min_pt[2]],
        [bbox.min_pt[0], bbox.max_pt[1], bbox.max_pt[2]],
        [bbox.max_pt[0], bbox.min_pt[1], bbox.min_pt[2]],
    ])

    def project(axis):
        return np.dot(corners - origin, axis)

    n_projs = project(se.n_vec)
    u_projs = project(se.u_vec)
    v_projs = project(se.v_vec)

    n_min, n_max = n_projs.min(), n_projs.max()
    u_min, u_max = u_projs.min(), u_projs.max()
    v_min, v_max = v_projs.min(), v_projs.max()

    B  = JB_OFFSET_BORDER
    T  = JB_OFFSET_THIN

    # Definitionen im lokalen KOS (relativ zu origin)
    # (n0, n1, u0, u1, v0, v1)
    jb_defs = {
        1: (n_min - T, n_max + T,  u_min - B, u_min + B,  v_min, v_max),
        2: (n_min - T, n_max + T,  u_min + B, u_max - B,  v_min, v_max),
        3: (n_min - T, n_max + T,  u_max - B, u_max + B,  v_min, v_max),
        4: (n_min - T, n_max + T,  u_min,     u_max,      v_min - B, v_min + B),
        5: (n_min - T, n_max + T,  u_min,     u_max,      v_min + B, v_max - B),
        6: (n_min - T, n_max + T,  u_min,     u_max,      v_max - B, v_max + B),
    }

    jbs: dict[int, JunctionBox] = {}
    for jb_id, (n0, n1, u0, u1, v0, v1) in jb_defs.items():
        world_bbox = build_world_bbox_from_local(
            n0, n1, u0, u1, v0, v1,
            se.n_vec, se.u_vec, se.v_vec,
            origin
        )
        jb = JunctionBox(box_id=jb_id, bbox=world_bbox, se=se)
        jbs[jb_id] = jb

    return jbs


# ===========================================================================
# SCHRITT 7 – FLANKIERENDE ELEMENTE DEN JUNCTION BOXES ZUWEISEN
# ===========================================================================

def determine_jb_id(se: BuildingElement,
                    fe: BuildingElement) -> Optional[int]:
    """
    Bestimmt EXAKT EINE JB-Nummer für ein FE.

    Strategie (rotationsrobust, eindeutig):
    1. Verbindungsvektor Zentrum->Zentrum im lokalen KOS des SE:
       [n-Anteil, u-Anteil, v-Anteil]
    2. Normiere auf die jeweilige SE-Ausdehnung -> vergleichbare Groessenordnung.
    3. Groesster normierter Anteil gewinnt. Bei Gleichstand: u > v > n.
    4. n-Dominant (paralleles Element): naechste Kante des SE entscheidet.
    """
    origin = se.bbox.center()
    delta  = fe.bbox.center() - origin

    n_comp = float(np.dot(delta, se.n_vec))
    u_comp = float(np.dot(delta, se.u_vec))
    v_comp = float(np.dot(delta, se.v_vec))

    # Ausdehnung des SE auf jeder lokalen Achse
    all_corners = np.array([
        se.bbox.min_pt, se.bbox.max_pt,
        [se.bbox.min_pt[0], se.bbox.max_pt[1], se.bbox.min_pt[2]],
        [se.bbox.max_pt[0], se.bbox.min_pt[1], se.bbox.max_pt[2]],
        [se.bbox.min_pt[0], se.bbox.min_pt[1], se.bbox.max_pt[2]],
        [se.bbox.max_pt[0], se.bbox.max_pt[1], se.bbox.min_pt[2]],
        [se.bbox.min_pt[0], se.bbox.max_pt[1], se.bbox.max_pt[2]],
        [se.bbox.max_pt[0], se.bbox.min_pt[1], se.bbox.min_pt[2]],
    ])
    n_projs  = np.dot(all_corners - origin, se.n_vec)
    u_projs  = np.dot(all_corners - origin, se.u_vec)
    v_projs  = np.dot(all_corners - origin, se.v_vec)
    n_extent = max(float(n_projs.max() - n_projs.min()), GEOM_TOL)
    u_extent = max(float(u_projs.max() - u_projs.min()), GEOM_TOL)
    v_extent = max(float(v_projs.max() - v_projs.min()), GEOM_TOL)
    u_min, u_max = float(u_projs.min()), float(u_projs.max())
    v_min, v_max = float(v_projs.min()), float(v_projs.max())

    # Normierte Betraege (dimensionslos, vergleichbar)
    abs_n = abs(n_comp) / n_extent
    abs_u = abs(u_comp) / u_extent
    abs_v = abs(v_comp) / v_extent

    log.debug(f"    DD-Normen  n={abs_n:.3f}  u={abs_u:.3f}  v={abs_v:.3f}"
              f"  (raw n={n_comp:.3f} u={u_comp:.3f} v={v_comp:.3f})")

    # Entscheidung: groesster normierter Anteil; Prioritaet u > v > n
    if abs_u >= abs_v and abs_u >= abs_n:
        return 1 if u_comp < 0 else 3

    elif abs_v >= abs_n:
        return 4 if v_comp < 0 else 6

    else:
        # n-Dominant: paralleles Element -> naechste Kante des SE
        u_fe = float(np.dot(fe.bbox.center() - origin, se.u_vec))
        v_fe = float(np.dot(fe.bbox.center() - origin, se.v_vec))
        B    = JB_OFFSET_BORDER
        dists = {
            1: abs(u_fe - u_min),
            3: abs(u_fe - u_max),
            4: abs(v_fe - v_min),
            6: abs(v_fe - v_max),
        }
        nearest = min(dists, key=lambda k: dists[k])
        if dists[nearest] > B:
            return 2 if u_extent >= v_extent else 5
        return nearest


def assign_flanking_to_jbs(se: BuildingElement,
                            flanking: list[BuildingElement],
                            jbs: dict[int, JunctionBox]) -> dict[int, JunctionBox]:
    """Weist jedes FE der korrekten JunctionBox zu."""
    for fe in flanking:
        jb_id = determine_jb_id(se, fe)
        if jb_id is None:
            log.warning(f"  FE #{fe.ifc_id} → keine JB ermittelt")
            continue
        added = jbs[jb_id].add_flanking(fe)
        if not added:
            log.warning(f"  JB{jb_id} voll – FE #{fe.ifc_id} verworfen")
        else:
            log.debug(f"  FE #{fe.ifc_id} ({fe.elem_direction}) → JB{jb_id}")
    return jbs


# ===========================================================================
# SCHRITT 8 – CONNECTION ZONES
# ===========================================================================

def get_connection_zone(host: BuildingElement,
                         visitor: BuildingElement) -> str:
    """
    Bestimmt, in welcher Zone das 'visitor'-Element auf das 'host'-Element trifft.
    Alle Berechnungen im lokalen KOS von 'host' → rotationsunabhängig.

    Zonen (nach Paper Abb. 5.39–5.41):
        "short"  – Stirnfläche (Dickenrichtung)
        "border" – Randstreifen auf der Hauptfläche (0.5 m breit)
        "middle" – Mittelfläche
    """
    origin = host.bbox.center()

    # Projektionsgrenzen des Host auf lokale Achsen
    corners = np.array([host.bbox.min_pt, host.bbox.max_pt])
    def proj_range(axis):
        p = np.dot(corners - origin, axis)
        return p.min(), p.max()

    n_min, n_max = proj_range(host.n_vec)
    u_min, u_max = proj_range(host.u_vec)
    v_min, v_max = proj_range(host.v_vec)

    # Nächster Punkt des Visitors zum Host (in Weltkoordinaten)
    contact_world = np.clip(visitor.bbox.center(),
                            host.bbox.min_pt, host.bbox.max_pt)
    delta = contact_world - origin

    c_n = float(np.dot(delta, host.n_vec))
    c_u = float(np.dot(delta, host.u_vec))
    c_v = float(np.dot(delta, host.v_vec))

    B = JB_OFFSET_BORDER

    # Auf der Stirnseite (short)?
    n_thick = n_max - n_min
    if abs(c_n - n_min) < n_thick * 0.15 or abs(c_n - n_max) < n_thick * 0.15:
        return "short"

    # Im Randbereich?
    on_u_border = (c_u < u_min + B or c_u > u_max - B)
    on_v_border = (c_v < v_min + B or c_v > v_max - B)
    if on_u_border or on_v_border:
        return "border"

    return "middle"


# ===========================================================================
# SCHRITT 9 – STOΒSTELLENTYP IDENTIFIZIEREN
# ===========================================================================

def _se_is_slab(se: BuildingElement) -> bool:
    """True wenn SE eine horizontal liegende Decke/Boden ist (n_vec zeigt stark in Z)."""
    return bool(abs(se.n_vec[2]) > ANGLE_TOL or se.ifc_type == "IfcSlab")


def identify_junction_type(jb: JunctionBox) -> tuple[str, str]:
    """
    Identifiziert den Stoßstellentyp aus einer Junction Box.

    h / v Unterscheidung (laut Paper):
      h = Trennelement ist eine Wand (vertical, n_vec liegt in XY-Ebene)
      v = Trennelement ist eine Decke/Boden (horizontal, n_vec zeigt in Z)

    JB-Nummer bedeutet:
      JB1/3 = Enden in u-Richtung (Wand: linkes/rechtes Ende; Decke: vorderes/hinteres Ende)
      JB2/5 = MITTE in u-Richtung → FE trifft SE in der Mitte → T-Stoß
      JB4/6 = Enden in v-Richtung (Wand: Oben/Unten; Decke: linkes/rechtes Ende)
    """
    se    = jb.se
    fes   = jb.flanking_elements()
    n     = len(fes)
    jb_id = jb.box_id
    slab  = _se_is_slab(se)   # SE ist Decke?

    if n == 0:
        return "NONE", "Keine flankierenden Elemente in dieser Box"

    dirs  = [fe.elem_direction for fe in fes]
    cz_fe = [get_connection_zone(se, fe) for fe in fes]
    cz_se = [get_connection_zone(fe, se) for fe in fes]

    log.debug(f"    identify_junction_type: jb_id={jb_id} n_fe={n} slab={slab} "
              f"dirs={dirs} cz_fe={cz_fe} cz_se={cz_se}")

    # --------------- 1 FE -----------------------------------------------
    if n == 1:
        d, czf, czs = dirs[0], cz_fe[0], cz_se[0]

        # JB2 / JB5: FE trifft SE in der MITTE → T-Stoß
        if jb_id in (2, 5):
            if slab:
                # SE = Decke, FE trifft Decke in Mitte
                if d in ("m", "n"):
                    return ("Tv2-13",
                            "T-Stoß: Wand trifft Decke(SE) in der Mitte (Tv2-13)")
                if d == "o":
                    return ("Tv2-1:3",
                            "T-Stoß: Decke trifft Decke(SE) in der Mitte (Tv2-1:3)")
            else:
                # SE = Wand, FE trifft Wand in Mitte
                if d == "m":
                    return ("Th1-24",
                            "T-Stoß horizontal: Querwand trifft SE-Wand in Mitte (Th1-24)")
                if d == "o":
                    return ("Tv1-24",
                            "T-Stoß vertikal: Decke trifft Wand(SE) in Mitte (Tv1-24)")
                if d == "n":
                    return ("Th1-24",
                            "T-Stoß horizontal: Parallelwand trifft SE-Wand in Mitte")

        # JB1 / JB3: Enden in u-Richtung
        elif jb_id in (1, 3):
            if slab:
                if d == "o":
                    return ("Lv1-2",   "L-Stoß: Decke–Decke, Ende")
                if d in ("m", "n"):
                    if czs == "middle":
                        return ("Tv2-13",  "T-Stoß: Wand trifft Decke(SE)-Mitte, Ende")
                    return ("Lv1-2",   "L-Stoß: Wand–Decke, Ende u-Richtung")
            else:
                if d == "o":
                    return ("Lv1-2",   "L-Stoß vertikal (Wand–Decke, Ende)")
                if d == "m":
                    if czs == "middle":
                        return ("Th1-24",  "T-Stoß horizontal: SE-Ende trifft Querwand-Mitte")
                    if czs == "border":
                        return ("Th1-2:4", "Th1-2:4: SE-Ende trifft Rand einer Querwand")
                    return ("Lh1-2",   "L-Stoß horizontal (Wand–Wand 90°, Ecke)")
                if d == "n":
                    # Parallele Wand → SE setzt sich fort
                    if czs in ("middle", "border"):
                        return ("Th1-2-4", "Th1-2-4: SE-Ende trifft parallele Wand (Rand/Mitte)")
                    return ("Lh1-2",   "L-Stoß horizontal (parallele Wände, Ende)")

        # JB4 / JB6: Enden in v-Richtung (Oben/Unten bei Wand; Seiten bei Decke)
        else:   # jb_id in (4, 6)
            if slab:
                # SE = Decke, Ende in v-Richtung (Seite der Decke)
                if d == "o":
                    return ("Lv1-2",   "L-Stoß: Decke–Decke, Seite")
                if d in ("m", "n"):
                    if czs == "middle":
                        return ("Tv2-13",  "T-Stoß: Wand trifft Decke(SE)-Mitte, Seite")
                    if czf == "border":
                        return ("Tv1-24",  "T-Stoß vertikal: Wand trifft Decke an Rand")
                    return ("Lv1-2",   "L-Stoß: Wand–Decke, Seite")
            else:
                # SE = Wand, Ende in v-Richtung (Oben/Unten)
                if d == "o":
                    if czs == "middle":
                        return ("Tv2-13",  "T-Stoß vertikal: Decke trifft Wand(SE)-Mitte")
                    if czf == "border":
                        return ("Tv1-24",  "T-Stoß vertikal: Decke trifft Wand an Rand")
                    return ("Lv1-2",   "L-Stoß vertikal (Wand–Decke)")
                if d == "m":
                    if czs in ("middle", "border"):
                        return ("Th1-24",  "T-Stoß horizontal: Querwand trifft SE Oben/Unten")
                    return ("Lh1-2",   "L-Stoß horizontal (Wand–Wand 90°)")
                if d == "n":
                    return ("Lv1-2",   "L-Stoß vertikal (parallele Wände)")

        # Fallback 1 FE
        prefix = "Tv" if (slab or d == "o") else "Th"
        return (f"{prefix}1-24", f"{prefix}-Stoß (1 FE, Fallback)")

    # --------------- 2 FE -----------------------------------------------
    elif n == 2:
        return _classify_2fe(se, fes, cz_fe, cz_se, dirs, jb_id, slab)

    # --------------- 3 FE -----------------------------------------------
    elif n == 3:
        return _classify_3fe(se, fes, cz_fe, cz_se, dirs, slab)

    else:
        return ("COMPLEX", f"Mehr als 3 FE in einer Box ({n} FE) – manuelle Prüfung")


def _classify_2fe(se, fes, cz_fe, cz_se, dirs, jb_id: int,
                  slab: bool) -> tuple[str, str]:
    """T-Stoß / X-Stoß für 2 FEs.

    Schlüsselunterscheidung anhand dd_local:
    ┌───────────────────┬─────────────────┬──────────────┐
    │ Typ               │ n-Seiten        │ v-Seiten     │
    ├───────────────────┼─────────────────┼──────────────┤
    │ Th1-24 / Tv1-24   │ GLEICH          │ beliebig     │
    │ Th2-1-4 / Tv2-1-4 │ GEGENÜBER       │ VERSCHIEDEN  │
    │ Th1-2:4 / Tv2-1:3 │ GEGENÜBER       │ GLEICH       │
    └───────────────────┴─────────────────┴──────────────┘
    """
    d0, d1   = dirs
    cz0, cz1 = cz_fe
    s0, s1   = cz_se
    dir_set  = set(dirs)

    def opposite_n_sides() -> bool:
        """FEs auf gegenüberliegenden n-Seiten (dd_local[0] entgegengesetzt)."""
        if len(fes) < 2:
            return False
        a = float(np.sign(fes[0].dd_local[0]))
        b = float(np.sign(fes[1].dd_local[0]))
        return a != 0 and b != 0 and a != b

    def same_v_side() -> bool:
        """FEs auf gleicher v-Seite (dd_local[2] gleiches Vorzeichen)."""
        if len(fes) < 2:
            return False
        a = fes[0].dd_local[2]
        b = fes[1].dd_local[2]
        # Beide nahe null → kein v-Versatz → gilt als gleiche Seite
        if abs(a) < GEOM_TOL and abs(b) < GEOM_TOL:
            return True
        sa, sb = float(np.sign(a)), float(np.sign(b))
        return sa != 0 and sb != 0 and sa == sb

    # ── JB2/5: FEs treffen SE in der MITTE ──
    if jb_id in (2, 5):
        if slab:
            # SE = Decke
            if dir_set <= {"m", "n"}:
                # Prüfe ob Wände auf gegenüberliegenden n-Seiten der Decke liegen
                opp = (
                    len(fes) == 2
                    and float(np.sign(fes[0].dd_local[0])) != 0
                    and float(np.sign(fes[1].dd_local[0])) != 0
                    and float(np.sign(fes[0].dd_local[0])) != float(np.sign(fes[1].dd_local[0]))
                )
                if opp:
                    return ("Xv2-13-4",
                            "Xv2-13-4: Decke(SE) in Mitte, Wände auf gegenüberl. n-Seiten")
                return ("Tv2-1:3",   "Tv2-1:3: 2 Wände treffen Decke(SE) in Mitte (gleiche Seite)")
            if "o" in dir_set:
                return ("Xh2-1:3-4", "X-Stoß horizontal gemischt (Wand+Decke in Decken-Mitte)")
        else:
            # SE = Wand
            # h = Kreuzung im Grundriss (alle vertikale Wände) → FEs sind Wände (m/n)
            # v = Kreuzung im Aufriss (Wand + Decken)          → FEs sind Decken (o)
            if dir_set <= {"m", "n"}:
                # 2 Querwände durch Wandmitte → Xh1-24-3
                return ("Xh1-24-3",  "Xh1-24-3: 2 Querwände durch Wandmitte (horizontal X)")
            if dir_set == {"o"}:
                # Xv1-24-3: Wand(SE) läuft durch, je eine Decke auf jeder n-Seite
                # Xv2-13-4: Decke(SE) in Mitte, Wände beidseitig → hier unmöglich (SE=Wand)
                # Unterscheidung: gegenüberliegende n-Seiten → Xv1-24-3
                opp = (
                    len(fes) == 2
                    and float(np.sign(fes[0].dd_local[0])) != 0
                    and float(np.sign(fes[1].dd_local[0])) != 0
                    and float(np.sign(fes[0].dd_local[0])) != float(np.sign(fes[1].dd_local[0]))
                )
                if opp:
                    return ("Xv1-24-3",  "Xv1-24-3: Wand(SE) durch Mitte, Decken auf gegenüberl. n-Seiten")
                return ("Xv2-13-4",      "Xv2-13-4: 2 Decken durch Wandmitte (gleiche n-Seite)")
            if "o" in dir_set:
                return ("Xh2-1:3-4", "X-Stoß horizontal gemischt (Wand+Decke)")
            return ("Xv2-1:3-4",     "X-Stoß vertikal gemischt")

    # ── Beide/gemischte Wände (m oder m+n) ──
    if dir_set <= {"m", "n"}:
        has_n = "n" in dir_set   # paralleles Element (SE-Fortsetzung)
        has_m = "m" in dir_set   # Querwand (90°)

        if cz0 == "short" and cz1 == "short":
            if opposite_n_sides():
                only_m = (dir_set == {"m"})
                only_n = (dir_set == {"n"})

                if only_m:
                    # 2 Querwände (m) auf gegenüberl. n-Seiten:
                    # SE=Decke → Tv2-1-4 (Decke zwischen zwei Wänden oben/unten)
                    # SE=Wand  → Th2-1-4 (Wand trennt zwei Querwände)
                    t = "Tv2-1-4" if slab else "Th2-1-4"
                    return (t, f"{t}: 2 m-Elemente auf gegenüberl. n-Seiten")

                if only_n:
                    # 2 parallele Wände (n) auf gegenüberl. n-Seiten:
                    # SE=Decke → Tv1-2:4 (Decke mit parallelen Wänden an Rand)
                    # SE=Wand  → Th1-2:4
                    t = "Tv1-2:4" if slab else "Th1-2:4"
                    return (t, f"{t}: 2 n-Elemente auf gegenüberl. n-Seiten")

                # Mischfall n+m: v-Seite entscheidet
                if same_v_side():
                    t = "Tv2-1-4" if slab else "Th1-2:4"
                    return (t, f"{t}: n+m, gleiche v-Seite, gegenüberl. n-Seiten")
                else:
                    t = "Tv1-2:4" if slab else "Th2-1-4"
                    return (t, f"{t}: n+m, versch. v-Seiten, gegenüberl. n-Seiten")

            t = "Tv1-24" if slab else "Th1-24"
            return (t, f"{t}: beidseitig short, gleiche n-Seite")

        if "middle" in (s0, s1):
            t = "Tv2-1-4" if slab else "Th2-1-4"
            return (t, f"{t}: SE-Ende trifft FE-Mitte")

        if "border" in (s0, s1):
            # Th1-2-4 / Tv1-2-4: ein n-Element (parallele Wand) + ein m-Element
            #   → Querwand liegt an der Übergangsstelle zweier fluchtender Wände
            # Th1-2:4 / Tv1-2:4: nur m-Elemente, SE-Ende liegt im Rand-CZ eines FEs
            #   → SE stößt gegen die Fläche einer Querwand, nicht an deren Ende
            if has_n and has_m:
                t = "Tv1-2-4" if slab else "Th1-2-4"
                return (t, f"{t}: paralleles(n) + Querwand(m), CZ=border")
            else:
                t = "Tv1-2:4" if slab else "Th1-2:4"
                return (t, f"{t}: nur Querwände(m), SE-Ende trifft FE-Rand")

        t = "Tv1-24" if slab else "Th1-24"
        return (t, f"{t}: allgemein")

    # ── Beide Decken (o) ──
    if dir_set == {"o"}:
        if cz0 == "short" and cz1 == "short":
            if opposite_n_sides():
                if same_v_side():
                    # Beide Decken am GLEICHEN Ende der Wand, auf gegenüberl. Wandflächen
                    return ("Tv2-1:3",
                            "Tv2-1:3: 2 Decken gleiche v-Seite, gegenüberl. n-Seiten der Wand")
                else:
                    # Eine Decke oben, eine unten — Wand trennt sie
                    return ("Tv2-1-4",
                            "Tv2-1-4: Wand(SE) trennt 2 Decken oben+unten (versch. v-Seiten)")
            return ("Tv1-24",  "Tv1-24: 2 Decken gleiche n-Seite")
        if "middle" in (s0, s1):
            return ("Tv2-1-4", "Tv2-1-4: SE-Ende trifft Decken-Mitte")
        return ("Tv1-2:4",     "Tv1-2:4: Decken mit Randzone")

    # ── Gemischt Wand + Decke ──
    if "o" in dir_set:
        # Tv2-1:3: SE=Decke, FEs = Wand(n, short) + Decke(o, border/short)
        # Wand berührt Decken-Stirnfläche; zweite Decke liegt in v-Richtung daneben.
        if slab and "n" in dir_set:
            n_idx = next((i for i, d in enumerate(dirs) if d == "n"), None)
            if n_idx is not None and cz_fe[n_idx] == "short":
                return ("Tv2-1:3",
                        "Tv2-1:3: SE-Decke + parallele Wand(n, short) + Nachbardecke(o)")
        if "middle" in cz_fe:
            return ("Tv2-13",  "Tv2-13: Decke trifft Wandmitte")
        return ("Tv1-24",      "Tv1-24: gemischt Wand+Decke")

    return ("Tv1-24", "T-Stoß (allgemein, 2 FE)")


def _classify_3fe(se, fes, cz_fe, cz_se, dirs, slab: bool) -> tuple[str, str]:
    """X-Stoß für 3 FEs.

    Konvention (konsistent mit 2-FE-Logik):
      h = Kreuzung im Grundriss, alle vertikale Wände  → FEs sind Wände (m/n)
      v = Kreuzung im Aufriss, Wände treffen Decken    → FEs enthalten Decken (o)
    """
    dc    = {d: dirs.count(d) for d in set(dirs)}
    o_cnt = dc.get("o", 0)
    m_cnt = dc.get("m", 0)
    n_cnt = dc.get("n", 0)

    if slab:
        # SE = Decke
        if m_cnt + n_cnt >= 2 and o_cnt == 1:
            return ("Xv2-1:3-4", "X-Stoß: Decke(SE) + 2 Wände + 1 Decke")
        if m_cnt + n_cnt == 3:
            return ("Xv2-13-4",  "X-Stoß: Decke(SE) + 3 Wände")
    else:
        # SE = Wand
        # Alle FEs sind Wände (m/n) → horizontales X im Grundriss
        if o_cnt == 0:
            return ("Xh1-24-3",  "Xh1-24-3: SE-Wand + 3 Querwände (horizontal X)")
        # FEs enthalten Decken → vertikales X im Aufriss
        if o_cnt >= 2:
            return ("Xv2-13-4",  "Xv2-13-4: SE-Wand + 2 Decken durch Mitte")
        if m_cnt + n_cnt >= 2 and o_cnt == 1:
            return ("Xh2-1:3-4", "Xh2-1:3-4: SE-Wand + 2 Wände + 1 Decke gemischt")
        if o_cnt == 1 and m_cnt == 0 and n_cnt == 2:
            return ("Xv1-24-3",  "Xv1-24-3: SE-Wand + 2 Parallelwände + 1 Decke")
    return ("Xv2-1:3-4", "X-Stoß vertikal gemischt (Fallback)")


# ===========================================================================
# SCHRITT 10 – HAUPTPIPELINE
# ===========================================================================

def run(ifc_path: str, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Logging einrichten ----
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

    # ---- Debug-Datenstruktur ----
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
        "elements":      [],
        "sep_elements":  [],
        "junction_boxes": [],
        "errors":        [],
    }
    results: dict = {
        "meta": {"ifc_path": str(ifc_path), "timestamp": ts},
        "summary": {},
        "junctions": [],
    }

    # ---- Pipeline ----
    model     = load_ifc(ifc_path)
    elements  = load_elements(model)
    debug["elements"] = [e.to_dict() for e in elements]

    sep_elems = select_separating_elements(elements, model)
    debug["sep_elements"] = [e.ifc_id for e in sep_elems]

    all_junctions: list[JunctionResult] = []

    for se in sep_elems:
        log.info(f"─── SE #{se.ifc_id} ({se.ifc_type}, {se.name}) ───")

        # Kandidaten: benachbarte Geschosse
        candidates = filter_by_storey(se, elements)

        # Flankierende Elemente
        flanking = find_flanking_elements(se, candidates, model)
        if not flanking:
            log.info(f"  Keine flankierenden Elemente → übersprungen")
            continue

        # Element-Directions
        assign_element_directions(se, flanking)

        # Junction Boxes aufbauen
        jbs = build_junction_boxes(se)

        # FE zuweisen
        jbs = assign_flanking_to_jbs(se, flanking, jbs)

        # Debug
        for jb in jbs.values():
            entry = jb.to_dict()
            entry["se_name"] = se.name
            debug["junction_boxes"].append(entry)

        # Stoßstellentyp je Box bestimmen
        for jb_id, jb in jbs.items():
            fes = jb.flanking_elements()
            if not fes:
                continue

            jtype, notes = identify_junction_type(jb)

            cz = {str(fe.ifc_id): get_connection_zone(se, fe) for fe in fes}
            ed = {str(fe.ifc_id): fe.elem_direction for fe in fes}

            confidence = "ok"
            if jtype in ("COMPLEX", "UNKNOWN"):
                confidence = "warn"

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

            log.info(f"  JB{jb_id}: {jtype:15s}  FEs: "
                     f"{[fe.ifc_id for fe in fes]}  [{confidence}]")
            if notes:
                log.debug(f"         Note: {notes}")

    # ---- Deduplizierung -----------------------------------------------
    # Phase 1: Gleiche Element-Mengen → nur die beste Perspektive behalten.
    # Phase 2: Teilmengen entfernen — eine 2-Element-Junction die vollständig
    #          in einer 3-Element-Junction enthalten ist wird verworfen.
    #          (z.B. {55,125} ist Teilmenge von {55,125,149} → verwerfen)
    JUNCTION_PRIO = {
        # T-Stöße und X-Stöße: spezifischste Typen, höchste Priorität
        "Th1-24": 0, "Th1-2:4": 0, "Th1-2-4": 0, "Th2-1-4": 0,
        "Tv1-24": 0, "Tv1-2:4": 0, "Tv1-2-4": 0, "Tv2-1-4": 0,
        "Tv2-13": 0, "Tv2-1:3": 0,
        "Xv1-24-3": 0, "Xv2-13-4": 0, "Xh1-24-3": 0,
        "Xh2-1:3-4": 0, "Xv2-1:3-4": 0,
        # L-Stöße: weniger spezifisch
        "Lh1-2": 1, "Lv1-2": 1,
        # Fallback
        "NONE": 9, "COMPLEX": 8, "UNKNOWN": 8,
    }
    CONF_PRIO = {"ok": 0, "warn": 1, "error": 2}

    # Phase 1: gleiche Menge → bestes Ergebnis behalten
    seen_keys: dict[frozenset, JunctionResult] = {}
    for jr in all_junctions:
        key = frozenset({jr.se_ifc_id} | set(jr.fe_ifc_ids))
        if key not in seen_keys:
            seen_keys[key] = jr
        else:
            existing = seen_keys[key]
            def score(r: JunctionResult) -> tuple:
                conf         = CONF_PRIO.get(r.confidence, 9)
                jprio        = JUNCTION_PRIO.get(r.junction_type, 5)
                middle_bonus = 0 if r.jb_id in (2, 5) else 1
                n_fe_bonus   = -len(r.fe_ifc_ids)
                jtype        = r.junction_type          # ← war fehlend → stiller Bug!
                slab_is_se   = "Slab" in r.se_type
                # Slab-SE bevorzugen für Tv* (außer Tv2-1:3) und Xv2*
                # Wand-SE bevorzugen für Th*, Xh*, Xv1-24-3, Tv2-1:3
                if (jtype.startswith("Tv") and jtype != "Tv2-1:3") or \
                   jtype in ("Xv2-13-4", "Xv2-1:3-4"):
                    slab_bonus = 0 if slab_is_se else 1
                else:
                    slab_bonus = 1 if slab_is_se else 0
                return (conf, jprio, middle_bonus, slab_bonus, n_fe_bonus)
            if score(jr) < score(existing):
                seen_keys[key] = jr

    # Phase 2: Teilmengen entfernen
    all_keys = list(seen_keys.keys())
    subset_keys = set()
    for i, ka in enumerate(all_keys):
        for j, kb in enumerate(all_keys):
            if i != j and ka < kb:          # ka ist echte Teilmenge von kb
                log.info(f"  Teilmenge entfernt: {set(ka)} ⊂ {set(kb)}")
                subset_keys.add(ka)

    deduped = [jr for key, jr in seen_keys.items() if key not in subset_keys]
    removed = len(all_junctions) - len(deduped)
    if removed:
        log.info(f"  Deduplizierung: {removed} Duplikat(e)/Teilmenge(n) entfernt "
                 f"→ {len(deduped)} eindeutige Stoßstellen")
    all_junctions = deduped
    # -------------------------------------------------------------------

    type_counts: dict[str, int] = {}
    for jr in all_junctions:
        type_counts[jr.junction_type] = type_counts.get(jr.junction_type, 0) + 1

    results["summary"] = {
        "total_junctions":   len(all_junctions),
        "separating_elements": len(sep_elems),
        "junction_type_counts": type_counts,
    }
    results["junctions"] = [jr.to_dict() for jr in all_junctions]

    # ---- Ausgabe schreiben ----
    def np_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Nicht serialisierbar: {type(obj)}")

    debug_path  = out / "debug_junction_analysis.json"
    result_path = out / "junctions_result.json"

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, ensure_ascii=False, default=np_encoder)

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
    # -----------------------------------------------------------------------
    # KONFIGURATION – hier IFC-Pfad und Ausgabeverzeichnis anpassen
    # -----------------------------------------------------------------------
    IFC_PATH = "./ifc-models/Lh1-2.ifc"
    OUT_DIR  = "./output"
    # -----------------------------------------------------------------------

    # Kommandozeilenargumente überschreiben die obigen Werte (optional)
    parser = argparse.ArgumentParser(
        description="Rotationsrobuste IFC-Stoßstellenanalyse (Junction Analysis)"
    )
    parser.add_argument("--ifc", default=IFC_PATH,
                        help=f"Pfad zur IFC-Datei (Standard: {IFC_PATH})")
    parser.add_argument("--out", default=OUT_DIR,
                        help=f"Ausgabeverzeichnis (Standard: {OUT_DIR})")
    args = parser.parse_args()

    ifc_path = pathlib.Path(args.ifc)
    if not ifc_path.exists():
        print(f"[FEHLER] IFC-Datei nicht gefunden: {ifc_path.resolve()}")
        print(f"         Bitte Pfad in der KONFIGURATION oben in main() anpassen.")
        sys.exit(1)

    run(str(ifc_path), args.out)


if __name__ == "__main__":
    main()
