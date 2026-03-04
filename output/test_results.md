# Junction Analysis – Testergebnisse

*2026-03-04 10:19:26*

**13 PASS** | **2 FAIL** | **0 SKIP**

| Status | Modell | Erwartet | Erkannt |
|--------|--------|----------|---------|
| ✅ PASS | `Lh1-2` | `Lh1-2` | `Lh1-2` |
| ✅ PASS | `Lv1-2` | `Lv1-2` | `Lv1-2` |
| ✅ PASS | `Th1-2-4` | `Th1-2-4` | `Th1-2-4` |
| ✅ PASS | `Th1-24` | `Th1-24` | `Th1-24` |
| ✅ PASS | `Th2-1-4` | `Th2-1-4` | `Th2-1-4` |
| ✅ PASS | `Tv1-2-4` | `Tv1-2:4` | `Tv1-2:4` |
| ✅ PASS | `Tv1-24` | `Tv1-24` | `Tv1-24` |
| ❌ FAIL | `Tv2-1-3` | `Tv2-1:3` | `Tv1-24` |
| ✅ PASS | `Tv2-1-4` | `Tv2-1-4` | `Tv2-1-4` |
| ✅ PASS | `Tv2-13` | `Tv2-13` | `Tv2-13` |
| ✅ PASS | `Xh1-24-3` | `Xh1-24-3` | `Xh1-24-3` |
| ❌ FAIL | `Xh2-1-3-4` | `Xh2-1:3-4` | `NONE` |
| ✅ PASS | `Xv1-24-3` | `Xv1-24-3` | `Xv1-24-3` |
| ✅ PASS | `Xv2-1-3-4` | `Xv2-1:3-4` | `Xv2-1:3-4` |
| ✅ PASS | `Xv2-13-4` | `Xv2-13-4` | `Xv2-13-4` |

---

## Fehler-Details

### Tv2-1-3
- **Erwartet:** `Tv2-1:3`
- **Erkannt:** `Tv1-24`
- **Debug-Datei:** `output/debug_FAIL_Tv2-1-3.json`

### Xh2-1-3-4
- **Erwartet:** `Xh2-1:3-4`
- **Erkannt:** `NONE`
- **Debug-Datei:** `output/debug_FAIL_Xh2-1-3-4.json`

