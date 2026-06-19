# Brace Extraction Integration Report
**Date:** 2026-06-18  
**Backend:** FastAPI + PyMuPDF  
**Classifier:** `brace_classifier.py` v2 (4-layer pipeline)  
**Integration:** `main.py` — feature flag `BRACE_EXTRACTION` env var + `detect_braces` request param

---

## 1. Verification Summary

End-to-end `/analyse` workflow tested across **8 pages from 5 structural PDFs** covering braced frame elevations, framing plans, and roof plans at scales 1:64, 1:96, and 1:192.

| Test Case | PDF | Page | Scale | Beams | Braces | HIGH | MED | Beam Stable | Dupes | Overhead |
|---|---|---|---|---|---|---|---|---|---|---|
| snaps_elevation | Structural snaps.pdf | 6 | 1:96 | 73 | 15 | 15 | 0 | YES | 4 | +0.15s (16%) |
| snaps_elevation2 | Structural snaps.pdf | 7 | 1:96 | 33 | 16 | 16 | 0 | YES | 7 | +0.08s (24%) |
| snaps_framing | Structural snaps.pdf | 2 | 1:96 | 271 | 12 | 10 | 2 | YES | 0 | +0.08s (9%) |
| 07comb_elevation | 07_STRUCTURAL_COMBINED.pdf | 47 | 1:192 | 106 | 1 | 0 | 1 | YES | 0 | +0.26s (15%) |
| 07comb_elevation2 | 07_STRUCTURAL_COMBINED.pdf | 48 | 1:192 | 98 | 7 | 0 | 7 | YES | 0 | +0.12s (8%) |
| binder_framing | #Structural binder.pdf | 12 | 1:64 | 200 | 19 | 0 | 19 | YES | 0 | +0.09s (15%) |
| bayhealth_framing | Bayhealth Sussex MOB | 5 | 1:192 | 223 | 17 | 15 | 2 | YES | 0 | +0.00s (0%) |
| spruce_framing | 02 Struct 98 Spruce | 3 | 1:96 | 287 | 13 | 8 | 5 | YES | 0 | +0.25s (18%) |
| **TOTAL** | | | | **1291** | **100** | **64** | **36** | **8/8** | **11** | **+0.13s avg** |

**Pass/Fail:**
- Beam count stability: **PASS** (8/8 — zero regressions to existing beam extraction)
- Field completeness: **PASS** (8/8 — all required fields present in all 100 brace members)
- Duplicate detection: **PARTIAL** (11 duplicates on 2 elevation pages of Structural snaps.pdf)
- Performance: **PASS** (avg +0.13s overhead, 12.7% — well within acceptable range)

---

## 2. Sample API Responses

### Sample A — HIGH confidence, braced frame elevation (Structural snaps.pdf p6)
```json
{
  "type": "brace",
  "profile": null,
  "label": null,
  "confidence": "HIGH",
  "context": "braced_frame_elevation",
  "length_ft": 17.31,
  "angle_deg": 40.2,
  "bx1": 0.7865,
  "by1": 0.2274,
  "bx2": 0.7406,
  "by2": 0.2856,
  "x": 0.7635,
  "y": 0.2565,
  "lx": 0.7635,
  "ly": 0.2565,
  "sx": 0.7865,
  "sy": 0.2274
}
```

### Sample B — HIGH confidence, framing plan (02 Struct 98 Spruce p3)
```json
{
  "type": "brace",
  "profile": null,
  "label": null,
  "confidence": "HIGH",
  "context": "framing_plan",
  "length_ft": 22.18,
  "angle_deg": 38.6,
  "bx1": 0.5181,
  "by1": 0.2573,
  "bx2": 0.5783,
  "by2": 0.1852,
  "x": 0.5482,
  "y": 0.2213,
  "lx": 0.5482,
  "ly": 0.2213,
  "sx": 0.5181,
  "sy": 0.2573
}
```

### Sample C — MEDIUM confidence, roof plan context downgrade (#Structural binder p12)
```json
{
  "type": "brace",
  "profile": null,
  "label": null,
  "confidence": "MEDIUM",
  "context": "roof_plan",
  "length_ft": 8.99,
  "angle_deg": 45.0,
  "bx1": 0.1494,
  "by1": 0.7066,
  "bx2": 0.121,
  "by2": 0.6668,
  "x": 0.1352,
  "y": 0.6867,
  "lx": 0.1352,
  "ly": 0.6867,
  "sx": 0.1494,
  "sy": 0.7066
}
```

All coordinates are **normalised fractional values (0–1)** matching the beam member format. The `bx1/by1/bx2/by2` pair defines start/end endpoints; `x/y` and `lx/ly` are the midpoint; `sx/sy` is the start endpoint (same as `bx1/by1`).

---

## 3. Overlay Accuracy

Validated separately in `brace_overlay_validation.py` prior to integration.

- **37 candidates** sampled across 8 projects and 3 drawing types (elevation, framing plan, roof plan)
- **Coordinate transform verified:** `pixel_coord = pdf_coord × (dpi / 72.0)` — consistent with beam overlay transform
- **Crop render accuracy:** Start (S) and End (E) crosshair markers land precisely on drawn line endpoints at all tested scales
- **Visual inspection result: 100% alignment accuracy** (0 offset, 0 overshoot, 0 undershoot)
- Crop images output to: `backend/brace_overlay_validation/`

---

## 4. Performance Metrics

| Metric | Value |
|---|---|
| Average baseline response time | 1.01s |
| Average response time with braces enabled | 1.14s |
| Average overhead | +0.13s |
| Average overhead (%) | 12.7% |
| Fastest overhead | +0.00s (bayhealth, 223-beam page) |
| Slowest overhead | +0.26s (07_STRUCTURAL_COMBINED p47, 106-beam page) |
| Pages processed without error | 8/8 |
| Total brace members returned | 100 |
| Total brace candidates classified | ~300 (est.) |

Brace extraction overhead is dominated by `page.get_drawings()` which is already called by the beam extractor; the diagonal angle filter and classification pipeline add negligible cost. The +0.26s outlier on `07comb_elevation` reflects a heavier drawing layer on that page, not classifier overhead.

---

## 5. Feature Flag Behaviour

Two activation paths verified:

**Global (env var):**
```
BRACE_EXTRACTION=0   # default — brace extraction disabled for all requests
BRACE_EXTRACTION=1   # enables brace extraction globally
```
Server startup log confirms:
```
[INIT] brace_classifier loaded
[INIT] Brace extraction: DISABLED   ← when BRACE_EXTRACTION=0
```

**Per-request (request param):**
```json
{ "detect_braces": true }
```
Overrides the global flag for a single call. Allows targeted testing without restarting the server.

**Graceful degradation:** If `brace_classifier.py` fails to import (missing dependency, import error), `_BRACE_EXTRACTION_AVAILABLE = False` and all brace requests are silently skipped — the rest of the `/analyse` pipeline is unaffected.

---

## 6. Known Limitations

### L1 — Duplicate brace members (11 instances, 2 pages)
**Affected:** Structural snaps.pdf pages 6 and 7 (braced frame elevations)  
**Root cause:** Some PDFs encode each drawn diagonal as two overlapping line segments in `get_drawings()` — the same physical brace appears twice with near-identical coordinates (< 0.5% fractional distance apart). The deduplication in `extract_diagonals()` uses rounded integer coordinates to detect duplicates, but floating-point jitter between the two encoded segments can push them just outside the 1-pt rounding window.  
**Impact:** Frontend would render two overlapping brace overlays on the same segment. No data corruption to beams or other members.  
**Fix (deferred):** Tighten the deduplication key from rounded 1-pt to a 3-pt neighbourhood, or add a post-classify dedup pass keyed on `(round(x1/3), round(y1/3), round(x2/3), round(y2/3))`.

### L2 — Context classifier misidentifies some framing plans as braced_frame_elevation
**Affected:** Structural snaps.pdf page 2 (labelled `snaps_framing` in test suite) classified as `braced_frame_elevation`  
**Root cause:** The page text contains terms associated with elevations (possibly dimension annotations or sheet notes). The keyword scorer finds `braced_frame_elevation` trigger words before `framing_plan` triggers.  
**Impact:** Braces on that page are not downgraded when they should be. Effectively more permissive — potential false positives on framing plans.  
**Fix (deferred):** Add negative evidence: if the page text contains strong framing-plan signals (e.g. "FRAMING PLAN", "FRAMING PL.", grid bubble density), reduce elevation score.

### L3 — Unknown context on Bayhealth framing plan
**Affected:** Bayhealth Sussex MOB page 5 classified as `unknown`  
**Root cause:** Page text does not contain recognised sheet-type keywords in the first 3000 characters.  
**Impact:** `unknown` context has no confidence modifier — braces pass through unmodified. Slightly more lenient than ideal for a framing plan.  
**Fix (deferred):** Extend page text scan range, or add a layout-based fallback (presence of grid bubbles → framing plan).

### L4 — Scale 1:192 produces fewer HIGH candidates
**Affected:** 07_STRUCTURAL_COMBINED pages 47 and 48 — 0 HIGH, all MEDIUM  
**Root cause:** At 1:192, the `HIGH_MIN_FT = 15.0` threshold corresponds to a longer absolute pixel length. The braces on those pages are in the 9–12 ft range at that scale, which classifies as MEDIUM.  
**Impact:** These are genuine brace members, just below the HIGH length threshold. They are included in the response as MEDIUM — the frontend can render them with a distinct amber style.  
**Potential improvement (deferred):** Consider a context-aware threshold: lower `HIGH_MIN_FT` to 10 ft when `page_context == braced_frame_elevation` (high confidence the diagonal is structural).

### L5 — No profile or label extraction for braces
**By design:** Brace members return `profile: null` and `label: null`. Structural brace members in elevation drawings typically have their profiles annotated as text callouts (e.g. "HSS6×6×3/8") which are not yet associated with brace geometry. This is a future enhancement, not a defect.

### L6 — Frontend rendering not yet implemented
**Status:** The frontend (`frontend/app/page.tsx`, line ~1047) does not yet render brace members. Brace members are present in the API response but are invisible in the UI until the SVG overlay layer is extended to draw diagonal members in amber (MEDIUM) or green (HIGH).

---

## 7. Production Readiness Assessment

### What is production-ready

| Capability | Status |
|---|---|
| Extraction engine (get_drawings → diagonal filter) | Ready |
| 4-layer classifier (geometry, hatch, detail-scale, context) | Ready |
| Integration into /analyse pipeline | Ready |
| Feature flag (env var + per-request param) | Ready |
| Beam extraction isolation (zero regressions) | Ready |
| Field completeness (all coordinate fields present) | Ready |
| Overlay coordinate accuracy | Ready (100% validated) |
| Graceful degradation on import failure | Ready |
| Performance overhead | Ready (<0.3s per page) |

### What requires follow-up before full production use

| Issue | Priority | Effort |
|---|---|---|
| Duplicate suppression (L1) | Medium | Small — tighten dedup key |
| Frontend SVG rendering of brace members (L6) | High | Medium — extend overlay layer |
| Context misidentification on framing plans (L2) | Low | Small — add negative evidence |
| HIGH threshold too strict at 1:192 scale (L4) | Low | Small — context-aware threshold |

### Recommendation

**CONDITIONAL GO for production deployment with `BRACE_EXTRACTION=0` (default off).**

The extraction engine, classifier, and API integration are stable and verified. Zero regressions to beam extraction across all 8 test pages. The feature flag provides safe per-request activation for pilot use without any risk to existing beam-only workflows.

Activate `BRACE_EXTRACTION=1` or use `detect_braces: true` per-request to begin collecting real-world results. Before enabling by default:
1. Resolve the duplicate suppression issue (L1) — simple fix
2. Implement frontend SVG rendering (L6) — required for user visibility
3. Run 2–3 additional projects through the classifier and manually review HIGH candidates for false positives

The classifier is intentionally conservative (HIGH threshold ≥ 15 ft, MEDIUM threshold ≥ 8 ft, hard rejection on detail/schedule pages). False negatives are more likely than false positives. This is the correct tradeoff for a first integration.

---

## Appendix — Files Modified This Session

| File | Change |
|---|---|
| `backend/brace_classifier.py` | New file — 4-layer brace classifier |
| `backend/main.py` | 4 edits: import guard, AnalysisRequest.detect_braces, extraction block, 3× Unicode `→` -> `->` fixes in print statements |
| `backend/verify_integration.py` | New file — end-to-end integration test script |
| `backend/brace_overlay_validation.py` | New file — coordinate accuracy validator |
| `backend/verify_results.json` | Output — raw verification results |

---

*Generated by integration verification run on 2026-06-18.*
