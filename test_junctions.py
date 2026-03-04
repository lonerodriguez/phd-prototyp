"""
=============================================================================
Junction Analysis – Batch-Test für alle Stoßstellentypen
=============================================================================
Ausführung:
    python test_junctions.py

Erwartet: IFC-Dateien in ./ifc-models/ mit Namen wie "Tv2-13.ifc" etc.
Ausgabe:  ./output/test_results.md   – Übersicht mit PASS/FAIL
          ./output/debug_FAIL_<typ>.json – Debug-Info nur bei Fehlern
=============================================================================
"""
import json
import pathlib
import subprocess
import sys
import datetime

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
IFC_DIR    = pathlib.Path("./ifc-models")
OUT_DIR    = pathlib.Path("./output")
SCRIPT     = pathlib.Path("./junction_analysis.py")

# Erwartete Ergebnisse: Dateiname (ohne .ifc) → erwarteter junction_type
# Varianten mit Bindestrich UND Doppelpunkt berücksichtigt
EXPECTED = {
    "Lh1-2":      "Lh1-2",
    "Lv1-2":      "Lv1-2",
    "Tv2-13":     "Tv2-13",
    "Th1-24":     "Th1-24",
    "Tv1-24":     "Tv1-24",
    "Th2-1-4":    "Th2-1-4",
    "Xh1-24-3":   "Xh1-24-3",
    "Tv2-1-3":    "Tv2-1:3",    # Dateiname mit -, Typ mit :
    "Th1-2-4":    "Th1-2-4",
    "Tv2-1-4":    "Tv2-1-4",
    "Tv1-2-4":    "Tv1-2:4",    # Dateiname mit -, Typ mit :
    "Xv1-24-3":   "Xv1-24-3",
    "Xv2-13-4":   "Xv2-13-4",
    "Xh2-1-3-4":  "Xh2-1:3-4", # Dateiname mit -, Typ mit :
    "Xv2-1-3-4":  "Xv2-1:3-4", # Dateiname mit -, Typ mit :
}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def run_analysis(ifc_path: pathlib.Path, out_subdir: pathlib.Path) -> dict | None:
    """Führt junction_analysis.py aus und gibt das Result-JSON zurück."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--ifc", str(ifc_path), "--out", str(out_subdir)],
            capture_output=True, text=True, timeout=120
        )
        result_file = out_subdir / "junctions_result.json"
        if result_file.exists():
            with open(result_file, encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        return {"error": str(e)}


def get_detected_type(result_json: dict) -> str | None:
    """Extrahiert den erkannten Stoßstellentyp aus dem Ergebnis."""
    if not result_json or "junctions" not in result_json:
        return None
    junctions = result_json.get("junctions", [])
    if not junctions:
        return None
    # Häufigster Typ (sollte bei Einzel-Modell genau einer sein)
    counts = result_json.get("summary", {}).get("junction_type_counts", {})
    if counts:
        return max(counts, key=counts.get)
    return junctions[0].get("junction_type")


def build_debug_info(model_name: str, expected: str, detected: str,
                     result_json: dict, debug_json: dict | None) -> dict:
    """Baut strukturierte Debug-Info für einen fehlgeschlagenen Test."""
    junctions = result_json.get("junctions", []) if result_json else []

    info = {
        "model":    model_name,
        "expected": expected,
        "detected": detected,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "all_junctions_before_dedup": [],
        "junction_boxes_with_elements": [],
        "elements": [],
        "analysis_hints": [],
    }

    # Alle gefundenen Junctions
    info["all_junctions_before_dedup"] = [
        {
            "se_id":    j.get("se_ifc_id"),
            "se_type":  j.get("se_type"),
            "jb_id":    j.get("jb_id"),
            "type":     j.get("junction_type"),
            "fe_ids":   j.get("fe_ifc_ids"),
            "fe_dirs":  j.get("element_directions"),
            "fe_cz":    j.get("connection_zones"),
            "notes":    j.get("notes"),
        }
        for j in junctions
    ]

    if debug_json:
        # Elemente
        info["elements"] = [
            {
                "id":    e.get("ifc_id"),
                "type":  e.get("ifc_type"),
                "n_vec": e.get("n_vec"),
                "u_vec": e.get("u_vec"),
                "v_vec": e.get("v_vec"),
                "dir":   e.get("elem_dir"),
                "bbox_size": [
                    round(e["bbox"]["max"][i] - e["bbox"]["min"][i], 4)
                    for i in range(3)
                ] if "bbox" in e else [],
            }
            for e in debug_json.get("elements", [])
        ]

        # Junction Boxes mit Elementen (nur nicht-leere)
        info["junction_boxes_with_elements"] = [
            {
                "se_id":   jb.get("se_id"),
                "box_id":  jb.get("box_id"),
                "fe_ids":  jb.get("fe_ids"),
                "fe_dirs": jb.get("fe_dirs"),
                "fe_dd":   jb.get("fe_dd"),
                "fe_dist": jb.get("fe_dist"),
            }
            for jb in debug_json.get("junction_boxes", [])
            if jb.get("fe_ids")
        ]

    # Analyse-Hinweise automatisch generieren
    hints = info["analysis_hints"]
    for jb in info["junction_boxes_with_elements"]:
        n_fes = len(jb["fe_ids"])
        dirs  = list(jb["fe_dirs"].values()) if jb["fe_dirs"] else []
        dd    = jb.get("fe_dd", {})

        # n-Seiten prüfen
        n_signs = []
        for fid, ddv in dd.items():
            if ddv and len(ddv) >= 1:
                import math
                s = math.copysign(1, ddv[0]) if abs(ddv[0]) > 1e-6 else 0
                n_signs.append(s)

        v_signs = []
        for fid, ddv in dd.items():
            if ddv and len(ddv) >= 3:
                import math
                s = math.copysign(1, ddv[2]) if abs(ddv[2]) > 1e-6 else 0
                v_signs.append(s)

        opp_n = len(set(n_signs)) > 1 and 0 not in set(n_signs)
        same_v = len(set(v_signs)) == 1

        hints.append({
            "se_id":    jb["se_id"],
            "jb_id":    jb["box_id"],
            "n_fe":     n_fes,
            "dirs":     dirs,
            "opposite_n_sides": opp_n,
            "same_v_side":      same_v,
            "n_signs":  n_signs,
            "v_signs":  v_signs,
        })

    return info


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    print(f"\n{'='*60}")
    print(f"Junction Analysis – Batch-Test")
    print(f"{'='*60}\n")

    for model_name, expected_type in sorted(EXPECTED.items()):
        ifc_file = IFC_DIR / f"{model_name}.ifc"

        if not ifc_file.exists():
            print(f"  SKIP  {model_name:20s}  (Datei nicht gefunden: {ifc_file})")
            results.append({"model": model_name, "status": "SKIP",
                            "expected": expected_type, "detected": "-"})
            skip_count += 1
            continue

        # Analyse ausführen
        out_sub = OUT_DIR / f"test_{model_name}"
        result_json = run_analysis(ifc_file, out_sub)
        detected = get_detected_type(result_json) or "NONE"

        # Normalisiere für Vergleich (- und : sind semantisch unterschiedlich, behalte)
        passed = (detected == expected_type)

        status = "PASS" if passed else "FAIL"
        icon   = "✓" if passed else "✗"
        print(f"  {icon} {status}  {model_name:20s}  erwartet={expected_type:15s}  erkannt={detected}")

        if passed:
            pass_count += 1
        else:
            fail_count += 1
            # Debug-Info laden
            debug_file = out_sub / "debug_junction_analysis.json"
            debug_json = None
            if debug_file.exists():
                with open(debug_file, encoding="utf-8") as f:
                    debug_json = json.load(f)

            debug_info = build_debug_info(
                model_name, expected_type, detected, result_json or {}, debug_json
            )

            debug_out = OUT_DIR / f"debug_FAIL_{model_name}.json"
            with open(debug_out, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            print(f"         → Debug: {debug_out}")

        results.append({
            "model":    model_name,
            "status":   status,
            "expected": expected_type,
            "detected": detected,
        })

    # ---------------------------------------------------------------------------
    # Markdown-Bericht
    # ---------------------------------------------------------------------------
    md_path = OUT_DIR / "test_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Junction Analysis – Testergebnisse\n\n")
        f.write(f"*{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"**{pass_count} PASS** | **{fail_count} FAIL** | **{skip_count} SKIP**\n\n")
        f.write("| Status | Modell | Erwartet | Erkannt |\n")
        f.write("|--------|--------|----------|---------|\n")
        for r in results:
            icon = "✅" if r["status"] == "PASS" else ("⏭️" if r["status"] == "SKIP" else "❌")
            f.write(f"| {icon} {r['status']} | `{r['model']}` | `{r['expected']}` | `{r['detected']}` |\n")
        f.write(f"\n---\n")
        if fail_count > 0:
            f.write(f"\n## Fehler-Details\n\n")
            for r in results:
                if r["status"] == "FAIL":
                    f.write(f"### {r['model']}\n")
                    f.write(f"- **Erwartet:** `{r['expected']}`\n")
                    f.write(f"- **Erkannt:** `{r['detected']}`\n")
                    f.write(f"- **Debug-Datei:** `output/debug_FAIL_{r['model']}.json`\n\n")

    print(f"\n{'='*60}")
    print(f"Ergebnis: {pass_count} PASS | {fail_count} FAIL | {skip_count} SKIP")
    print(f"Bericht:  {md_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
