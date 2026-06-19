"""
Brace Confidence Classifier — Universal Multi-Project Detector
---------------------------------------------------------------
Classifies diagonal line candidates into HIGH / MEDIUM / REJECT with four
layers of filtering:

  Layer 1 — Geometry
    HIGH      length 15–200 ft  AND  angle 20–70°  AND  not density-cluster
    MEDIUM    length  8–15 ft   AND  angle 20–70°  AND  not density-cluster
    REJECT    density-cluster / too-short / too-long

  Layer 2 — Hatch Boundary
    REJECT if ≥ HATCH_MIN_PARALLEL shorter parallel lines exist within the
    candidate's bounding box (corner-to-corner diagonal of a hatched region).

  Layer 3 — Detail Scale Protection
    For each candidate, find the nearest scale annotation on the page.
    If a local scale is found that is ≥ 2× larger than the page scale AND the
    candidate's length at that local scale is < MEDIUM_MIN_FT → REJECT as
    "detail_scale".

  Layer 4 — Drawing Context
    Classify each page as one of:
      braced_frame_elevation, framing_plan, roof_plan, foundation_plan,
      detail, schedule_legend, unknown

    Context effects:
      braced_frame_elevation → no change (highest trust)
      framing_plan           → no change
      roof_plan              → downgrade HIGH → MEDIUM (real braces kept visible)
      foundation_plan        → tag only, no confidence change
      detail                 → HARD REJECT all candidates
      schedule_legend        → HARD REJECT all candidates
      unknown                → no change

Output
------
  • Console table per PDF + per page
  • JSON: brace_classifier_results.json  (per-page counts + context labels)
  • Per-page PNG overlays: HIGH=green  MEDIUM=amber  REJECT=grey

Usage
-----
  python brace_classifier.py               # all PDFs in uploads/
  python brace_classifier.py --pages       # also print per-page detail
"""
import os, sys, re, math, json, argparse
import fitz
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUT_DIR    = os.path.join(HERE, "brace_classifier_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tunable thresholds ────────────────────────────────────────────────────────
ANGLE_MIN          = 20.0   # degrees from horizontal
ANGLE_MAX          = 70.0
HIGH_MIN_FT        = 15.0
MEDIUM_MIN_FT      =  8.0
MAX_LENGTH_FT      = 200.0  # reject impossibly long lines (building boundaries, etc.)
DENSITY_BIN_FT     =  1.0   # bucket width for cluster detection
DENSITY_THRESHOLD  =  8     # candidates per bucket → cluster
HATCH_MIN_PARALLEL =  6     # min parallel shorter lines within bbox -> hatch boundary
DETAIL_SCALE_RATIO =  2.0   # local scale must be >= this x page scale to trigger
SCALE_SEARCH_DIST  = 400.0  # pts radius to search for a local scale annotation
DEDUP_PT           = 15.0   # per-endpoint pt tolerance for second-pass proximity dedup
                             # catches BIM-export pattern: same brace in two PDF path
                             # objects with ~6-12 pt endpoint jitter (measured max: 12.5 pt)

# Context hard-reject: pages of these types cannot contain structural braces
HARD_REJECT_CONTEXTS = {"detail", "schedule_legend"}

# ── PDFs to scan ──────────────────────────────────────────────────────────────
SCAN_PDFS = [
    ("#Structural binder.pdf",                               64.0),
    ("Structural snaps.pdf",                                 96.0),
    ("07_STRUCTURAL_COMBINED.pdf",                          192.0),
    ("Latest_Structural dwg_Binder (Addendum-02).pdf",       96.0),
    ("04_-_STRUCTURAL.pdf",                                  96.0),
    ("2026.03.27_Bayhealth Sussex MOB_DD Set_Structural.pdf",192.0),
    ("NCU SherMan_Structural.pdf",                           96.0),
    ("STRUCTURAL 5-26-26.pdf",                               96.0),
    ("Pages from 2026.05.08_FF Martha Washington Building - Issued for Pricing.pdf", 96.0),
    ("02 Struct 98 Spruce_2026-03-13_BID.pdf",               96.0),
]


def scale_to_pts_per_foot(scale_ratio: float) -> float:
    return 864.0 / scale_ratio if scale_ratio > 0 else 9.0


def _dedup_proximity(segments: list) -> list:
    """
    Second-pass dedup: remove segments that are geometrically near an
    already-kept longer segment.

    Catches the BIM-export pattern where the same structural brace is encoded
    as two separate PDF path objects (different drawing_idx) with consistent
    endpoint jitter of 6-12 pt.  The first-pass key-based dedup uses a 1-pt
    bin and cannot catch this.

    Algorithm: sort longest-first (keep the authoritative member); for each
    candidate check if the sum of both endpoint distances to any kept segment
    is < DEDUP_PT * 2 (forward or reversed).  If so, suppress the candidate.

    Safety: genuinely adjacent braces share at most ONE near endpoint (the
    shared column joint); their far endpoints are a full bay width apart,
    making the total endpoint-pair distance >> DEDUP_PT * 2.
    """
    segs = sorted(segments, key=lambda s: s["length_ft"], reverse=True)
    kept = []
    for s in segs:
        is_dup = False
        for k in kept:
            dfwd = (math.hypot(s["x1"] - k["x1"], s["y1"] - k["y1"]) +
                    math.hypot(s["x2"] - k["x2"], s["y2"] - k["y2"]))
            drev = (math.hypot(s["x1"] - k["x2"], s["y1"] - k["y2"]) +
                    math.hypot(s["x2"] - k["x1"], s["y2"] - k["y1"]))
            if min(dfwd, drev) < DEDUP_PT * 2:
                is_dup = True
                break
        if not is_dup:
            kept.append(s)
    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — Geometry extraction + density cluster
# ══════════════════════════════════════════════════════════════════════════════

def extract_diagonals(page, ppf: float) -> list:
    """
    Return every solid, non-dashed diagonal (20–70°) line segment ≥ 3 ft on
    the page.  No upper length limit here — filters applied in classify().
    """
    min_pt   = 3.0 * ppf
    results  = []
    seen     = set()

    try:
        for d in page.get_drawings():
            dashes = str(d.get("dashes") or "").strip()
            da = re.match(r'\[([^\]]*)\]', dashes)
            if da and da.group(1).strip():
                continue

            sw = float(d.get("width") or 0)

            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                try:
                    p1, p2 = item[1], item[2]
                    dx, dy = p2.x - p1.x, p2.y - p1.y
                    ln     = math.hypot(dx, dy)
                    if ln < min_pt:
                        continue

                    ang   = abs(math.degrees(math.atan2(dy, dx))) % 180
                    ang_h = min(ang, 180 - ang)
                    if not (ANGLE_MIN <= ang_h <= ANGLE_MAX):
                        continue

                    key  = (round(p1.x), round(p1.y), round(p2.x), round(p2.y))
                    rkey = (round(p2.x), round(p2.y), round(p1.x), round(p1.y))
                    if key in seen or rkey in seen:
                        continue
                    seen.add(key)

                    results.append({
                        "x1":          p1.x,
                        "y1":          p1.y,
                        "x2":          p2.x,
                        "y2":          p2.y,
                        "length_ft":   round(ln / ppf, 2),
                        "angle_from_h":round(ang_h, 1),
                        "stroke_w":    round(sw, 2),
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return _dedup_proximity(results)


def find_density_clusters(candidates: list) -> set:
    """
    Bin candidates by floor(length_ft / DENSITY_BIN_FT).
    Return indices whose bin has ≥ DENSITY_THRESHOLD members.
    """
    bins: dict[int, list] = defaultdict(list)
    for i, c in enumerate(candidates):
        b = int(c["length_ft"] / DENSITY_BIN_FT)
        bins[b].append(i)

    cluster = set()
    for idx_list in bins.values():
        if len(idx_list) >= DENSITY_THRESHOLD:
            cluster.update(idx_list)
    return cluster


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — Hatch boundary detection
# ══════════════════════════════════════════════════════════════════════════════

def is_hatch_boundary(candidate: dict, all_diagonals: list) -> bool:
    """
    Returns True if the candidate is the corner-to-corner diagonal of a
    hatched fill region.

    Detects: ≥ HATCH_MIN_PARALLEL lines that are (a) at the same angle ±5°,
    (b) shorter than the candidate, and (c) have their midpoint inside the
    candidate's bounding box.
    """
    c_ang = candidate["angle_from_h"]
    c_len = candidate["length_ft"]
    bx0   = min(candidate["x1"], candidate["x2"])
    by0   = min(candidate["y1"], candidate["y2"])
    bx1   = max(candidate["x1"], candidate["x2"])
    by1   = max(candidate["y1"], candidate["y2"])
    pad   = 20.0

    count = 0
    for line in all_diagonals:
        if line is candidate:
            continue
        if abs(line["angle_from_h"] - c_ang) > 5.0:
            continue
        if line["length_ft"] >= c_len * 0.95:
            continue
        mx = (line["x1"] + line["x2"]) / 2
        my = (line["y1"] + line["y2"]) / 2
        if (bx0 - pad <= mx <= bx1 + pad and
                by0 - pad <= my <= by1 + pad):
            count += 1
            if count >= HATCH_MIN_PARALLEL:
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — Detail scale protection
# ══════════════════════════════════════════════════════════════════════════════

# Regex: N/D" = 1'-0"  or  N" = 1'-0"  (architectural scale annotations)
_SCALE_FRAC = re.compile(
    r'(\d{1,3})\s*/\s*(\d{1,3})\s*["\']?\s*=\s*1\s*[-\']',
    re.I)
_SCALE_INT  = re.compile(
    r'(\d{1,2}(?:\.\d)?)\s*["\']?\s*=\s*1\s*[-\']',
    re.I)


def find_scale_annotations(page) -> list:
    """
    Parse every text span on the page for architectural scale annotations.
    Returns list of (ppf_at_this_scale, x, y).

    Formula:  ppf = 72 × (N/D)  for "N/D" = 1'" pattern
              ppf = 72 × N       for "N" = 1'" pattern
    (72 pts/inch on-screen; N/D inches on drawing = 1 foot in reality)
    """
    annotations = []
    try:
        td = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    except Exception:
        return annotations

    for block in td.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if not t or "=" not in t:
                    continue
                ox = span.get("origin", (0, 0))[0]
                oy = span.get("origin", (0, 0))[1]

                m = _SCALE_FRAC.search(t)
                if m:
                    num, den = int(m.group(1)), int(m.group(2))
                    if num > 0 and den > 0:
                        ppf = 72.0 * num / den
                        annotations.append((ppf, ox, oy))
                    continue

                m = _SCALE_INT.search(t)
                if m:
                    n = float(m.group(1))
                    if 0 < n <= 12:   # sanity: 1"=1' to 12"=1'
                        ppf = 72.0 * n
                        annotations.append((ppf, ox, oy))

    # Deduplicate by rounding ppf to 1 decimal
    seen_ppf = set()
    unique = []
    for (ppf, x, y) in annotations:
        key = round(ppf, 1)
        if key not in seen_ppf:
            seen_ppf.add(key)
            unique.append((ppf, x, y))
    return unique


def _nearest_scale(candidate: dict, scale_annotations: list, page_ppf: float):
    """
    Return the (local_ppf, dist_pts) of the nearest scale annotation within
    SCALE_SEARCH_DIST that has ppf > page_ppf * DETAIL_SCALE_RATIO.
    Returns (None, None) if nothing qualifies.
    """
    mx = (candidate["x1"] + candidate["x2"]) / 2
    my = (candidate["y1"] + candidate["y2"]) / 2

    best_ppf  = None
    best_dist = float("inf")
    for (ann_ppf, ax, ay) in scale_annotations:
        if ann_ppf < page_ppf * DETAIL_SCALE_RATIO:
            continue          # not a larger-scale detail
        dist = math.hypot(mx - ax, my - ay)
        if dist < best_dist and dist <= SCALE_SEARCH_DIST:
            best_dist = dist
            best_ppf  = ann_ppf

    return (best_ppf, best_dist) if best_ppf is not None else (None, None)


def check_detail_scale(candidate: dict, scale_annotations: list,
                       page_ppf: float) -> tuple:
    """
    Returns (is_mismatch: bool, reason: str|None).

    If a nearby detail annotation implies a much larger scale, re-compute the
    candidate's length at the local scale.  If that real length < MEDIUM_MIN_FT
    the candidate is a scale-mismatch false positive.
    """
    local_ppf, _ = _nearest_scale(candidate, scale_annotations, page_ppf)
    if local_ppf is None:
        return False, None

    real_ft = candidate["length_ft"] * page_ppf / local_ppf
    if real_ft < MEDIUM_MIN_FT:
        return True, f"detail_scale({local_ppf:.1f}pt/ft→{real_ft:.1f}ft)"
    return False, None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — Drawing context classification
# ══════════════════════════════════════════════════════════════════════════════

# (pattern, weight) pairs per context type
_CTX_PATTERNS: dict[str, list] = {
    "braced_frame_elevation": [
        (re.compile(r'BRACED\s+FRAME', re.I), 3),
        (re.compile(r'BRAC(?:ED|ING)\s+(?:FRAME\s+)?ELEVATION', re.I), 3),
        (re.compile(r'\bBF\s*[-–]\s*\d+\b', re.I), 2),
        (re.compile(r'BRACING\s+ELEVATION', re.I), 3),
        (re.compile(r'LATERAL\s+FRAME\s+ELEVATION', re.I), 3),
    ],
    "framing_plan": [
        (re.compile(r'FRAMING\s+PLAN', re.I), 3),
        (re.compile(r'LEVEL\s+\d+\s+FRAMING', re.I), 3),
        (re.compile(r'STRUCTURAL\s+(?:FLOOR\s+)?PLAN', re.I), 2),
        (re.compile(r'PARTIAL\s+FRAMING\s+PLAN', re.I), 3),
        (re.compile(r'FLOOR\s+PLAN(?!\s+OVERALL)', re.I), 2),
    ],
    "roof_plan": [
        (re.compile(r'ROOF\s+FRAMING\s+PLAN', re.I), 3),
        (re.compile(r'ROOF\s+PLAN', re.I), 3),
        (re.compile(r'ROOF\s+DECK\s+PLAN', re.I), 3),
        (re.compile(r'(?:HIGH|LOW|OVERALL)\s+ROOF', re.I), 3),
        (re.compile(r'CANOPY\s+(?:FRAMING\s+)?PLAN', re.I), 2),
        (re.compile(r'ROOF\s+JOIST', re.I), 2),
    ],
    "foundation_plan": [
        (re.compile(r'FOUNDATION\s+PLAN', re.I), 3),
        (re.compile(r'MAT\s+FOUNDATION', re.I), 2),
        (re.compile(r'FOOTING\s+PLAN', re.I), 3),
        (re.compile(r'PILE\s+(?:LAYOUT\s+)?PLAN', re.I), 2),
        (re.compile(r'GRADE\s+BEAM\s+PLAN', re.I), 2),
    ],
    "detail": [
        (re.compile(r'JOIST\s+POINT\s+LOAD\s+DIAGRAM', re.I), 4),
        (re.compile(r'PARTIAL\s+(?:ROOF\s+)?JOIST', re.I), 3),
        (re.compile(r'CONNECTION\s+DETAIL', re.I), 2),
        (re.compile(r'STUD\s+WALL\s+DETAIL', re.I), 2),
        (re.compile(r'NOT\s+TO\s+SCALE\b|\bNTS\b', re.I), 2),
        (re.compile(r'SECTION\s+[A-Z]-[A-Z]', re.I), 2),
        (re.compile(r'WELD\s+SCHEDULE|BOLT\s+SCHEDULE', re.I), 2),
    ],
    "schedule_legend": [
        (re.compile(r'\bSCHEDULE\b', re.I), 2),
        (re.compile(r'\bLEGEND\b', re.I), 2),
        (re.compile(r'GENERAL\s+NOTES', re.I), 2),
        (re.compile(r'\bKEYNOTES\b', re.I), 2),
        (re.compile(r'ABBREVIATION', re.I), 2),
        (re.compile(r'TYPICAL\s+NOTES', re.I), 2),
        (re.compile(r'TYPICAL\s+BRAC(?:ED?|ING)\s+FRAME', re.I), 4),
        (re.compile(r'FOR\s+TYPICAL\s+BRAC', re.I), 4),
    ],
}

# Confidence modifier per context type
CONTEXT_MODIFIER = {
    "braced_frame_elevation": +1,   # upgrade: MEDIUM→HIGH if length qualifies
    "framing_plan":            0,
    "roof_plan":              -1,   # downgrade: HIGH→MEDIUM
    "foundation_plan":         0,   # tag only
    "detail":                 -99,  # HARD REJECT
    "schedule_legend":        -99,  # HARD REJECT
    "unknown":                 0,
}


def classify_page_context(page) -> tuple:
    """
    Classify page type from the first ~3000 chars of page text.
    Returns (context_type: str, score: int).
    """
    try:
        text = page.get_text("text")[:3000]
    except Exception:
        return "unknown", 0

    scores: dict[str, int] = defaultdict(int)
    for ctx_type, patterns in _CTX_PATTERNS.items():
        for pat, weight in patterns:
            if pat.search(text):
                scores[ctx_type] += weight

    if not scores:
        return "unknown", 0

    # ── Multi-frame brace schedule detection ─────────────────────────────────
    # A placement drawing has at most 1-2 distinct BF-X labels (the frames drawn
    # on that sheet).  A brace type schedule packs BF-1, BF-2, BF-3 … on one
    # page to show all frame configurations.  3+ unique BF-X labels is an
    # unambiguous signal: this is a reference schedule, not a placement drawing.
    _BF_LABEL_RE = re.compile(r'BF\s*[-–]\s*\d+', re.I)
    unique_bf_labels = set(m.group().upper().replace(' ', '').replace('–', '-')
                           for m in _BF_LABEL_RE.finditer(text))
    if len(unique_bf_labels) >= 3:
        return "schedule_legend", 99

    # Resolve ties: elevation > framing_plan > roof_plan > others
    priority = ["braced_frame_elevation", "framing_plan", "roof_plan",
                "foundation_plan", "detail", "schedule_legend"]
    best_score = max(scores.values())
    best_types = [t for t in priority if scores.get(t, 0) == best_score]
    return best_types[0], best_score


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFY — combine all four layers
# ══════════════════════════════════════════════════════════════════════════════

def classify(candidates: list,
             page_context: str = "unknown",
             scale_annotations: list = None,
             ppf: float = 9.0) -> list:
    """
    Apply four-layer classification to every candidate.

    Parameters
    ----------
    candidates        : output of extract_diagonals()
    page_context      : output of classify_page_context()[0]
    scale_annotations : output of find_scale_annotations()
    ppf               : pts-per-foot for this page

    Added keys per candidate:
      confidence    : "HIGH" | "MEDIUM" | "REJECT"
      reject_reason : str | None
      page_context  : str (propagated metadata)
    """
    cluster_set = find_density_clusters(candidates)
    modifier    = CONTEXT_MODIFIER.get(page_context, 0)
    results     = []

    for i, c in enumerate(candidates):
        c = dict(c)
        c["page_context"] = page_context

        # ── Layer 1: geometry ────────────────────────────────────────────────
        if i in cluster_set:
            c["confidence"]    = "REJECT"
            c["reject_reason"] = "density_cluster"
            results.append(c)
            continue

        if c["length_ft"] < MEDIUM_MIN_FT:
            c["confidence"]    = "REJECT"
            c["reject_reason"] = "too_short"
            results.append(c)
            continue

        if c["length_ft"] > MAX_LENGTH_FT:
            c["confidence"]    = "REJECT"
            c["reject_reason"] = "too_long"
            results.append(c)
            continue

        # ── Layer 4 (context) hard-reject: check before geometry work ────────
        if modifier <= -99:
            c["confidence"]    = "REJECT"
            c["reject_reason"] = f"context:{page_context}"
            results.append(c)
            continue

        # Base confidence from length
        base = "HIGH" if c["length_ft"] >= HIGH_MIN_FT else "MEDIUM"

        # ── Layer 2: hatch boundary ──────────────────────────────────────────
        if is_hatch_boundary(c, candidates):
            c["confidence"]    = "REJECT"
            c["reject_reason"] = "hatch_boundary"
            results.append(c)
            continue

        # ── Layer 3: detail scale mismatch ───────────────────────────────────
        if scale_annotations:
            is_mm, mm_reason = check_detail_scale(c, scale_annotations, ppf)
            if is_mm:
                c["confidence"]    = "REJECT"
                c["reject_reason"] = mm_reason
                results.append(c)
                continue

        # ── Layer 4: context modifier (soft adjustment) ───────────────────────
        if modifier < 0 and base == "HIGH":
            base = "MEDIUM"
            c["reject_reason"] = None
        elif modifier > 0 and base == "MEDIUM" and c["length_ft"] >= HIGH_MIN_FT:
            base = "HIGH"
            c["reject_reason"] = None
        else:
            c["reject_reason"] = None

        c["confidence"] = base
        results.append(c)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def page_metrics(classified: list, context: str = "unknown") -> dict:
    high   = [c for c in classified if c["confidence"] == "HIGH"]
    medium = [c for c in classified if c["confidence"] == "MEDIUM"]
    reject = [c for c in classified if c["confidence"] == "REJECT"]
    reasons: dict[str, int] = defaultdict(int)
    for c in reject:
        reasons[c.get("reject_reason") or "unknown"] += 1
    return {
        "total":          len(classified),
        "high":           len(high),
        "medium":         len(medium),
        "reject":         len(reject),
        "reject_reasons": dict(reasons),
        "high_lengths":   sorted([c["length_ft"] for c in high],   reverse=True),
        "medium_lengths": sorted([c["length_ft"] for c in medium], reverse=True),
        "context":        context,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAY RENDERER
# ══════════════════════════════════════════════════════════════════════════════

_CONF_COLOR = {
    "HIGH":   (50,  205,  50),   # green
    "MEDIUM": (255, 195,   0),   # amber
    "REJECT": (100, 100, 100),   # grey
}

# Reject-reason colour overrides (shown instead of plain grey for key FP types)
_REJECT_COLORS = {
    "hatch_boundary":  (200,  80, 200),   # magenta
    "too_long":        (255,  80,  80),   # red
    "too_short":       ( 80,  80,  80),   # dark grey (same as reject)
    "density_cluster": ( 80,  80,  80),
}


def render_classified_overlay(page, classified, ppf, title, dpi=100):
    pix   = page.get_pixmap(dpi=dpi)
    img   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw  = ImageDraw.Draw(img)
    scale = dpi / 72.0

    try:
        font = ImageFont.truetype("arial.ttf", 9)
    except Exception:
        font = ImageFont.load_default()

    # Draw REJECT first (bottom), then MEDIUM, then HIGH on top
    for conf in ("REJECT", "MEDIUM", "HIGH"):
        for c in classified:
            if c["confidence"] != conf:
                continue
            x1 = c["x1"] * scale;  y1 = c["y1"] * scale
            x2 = c["x2"] * scale;  y2 = c["y2"] * scale

            if conf == "REJECT":
                rr  = c.get("reject_reason") or ""
                col = _REJECT_COLORS.get(rr, _CONF_COLOR["REJECT"])
                w   = 2 if rr in ("hatch_boundary", "too_long") else 1
            else:
                col = _CONF_COLOR[conf]
                w   = 3 if conf == "HIGH" else 2

            draw.line([(x1, y1), (x2, y2)], fill=col, width=w)

            if conf != "REJECT":
                r = 3
                draw.ellipse([(x1-r, y1-r), (x1+r, y1+r)], fill=col)
                draw.ellipse([(x2-r, y2-r), (x2+r, y2+r)], fill=col)

            if conf == "HIGH":
                mx = (x1 + x2) / 2;  my = (y1 + y2) / 2
                txt = f"{c['length_ft']:.0f}ft"
                try:
                    bb = draw.textbbox((mx, my), txt, font=font)
                    draw.rectangle(bb, fill=(0, 0, 0))
                except Exception:
                    pass
                draw.text((mx, my), txt, fill=(50, 255, 50), font=font)

    # Context label in top-right
    ctx = classified[0]["page_context"] if classified else "unknown"
    ctx_color = {
        "braced_frame_elevation": (50, 255, 50),
        "framing_plan":           (100, 200, 255),
        "roof_plan":              (255, 200, 80),
        "foundation_plan":        (200, 160, 80),
        "detail":                 (255, 80,  80),
        "schedule_legend":        (255, 80,  80),
    }.get(ctx, (180, 180, 180))
    ctx_w = 160
    draw.rectangle([(img.width - ctx_w - 4, 4),
                    (img.width - 4, 18)], fill=(20, 20, 20))
    draw.text((img.width - ctx_w - 2, 5), ctx, fill=ctx_color, font=font)

    # Legend
    legend_items = [
        (_CONF_COLOR["HIGH"],   "HIGH   (>=15ft)"),
        (_CONF_COLOR["MEDIUM"], "MEDIUM (8-15ft, or roof plan)"),
        (_CONF_COLOR["REJECT"], "REJECT (cluster/short/long/context)"),
        (_REJECT_COLORS["hatch_boundary"], "REJECT:hatch_boundary"),
    ]
    ly = 6
    draw.rectangle([(4, 3), (210, 6 + len(legend_items) * 14 + 4)],
                   fill=(15, 15, 15))
    for col, txt in legend_items:
        draw.rectangle([(7, ly), (20, ly + 9)], fill=col)
        draw.text((23, ly), txt, fill=(210, 210, 210), font=font)
        ly += 13

    high_c   = sum(1 for c in classified if c["confidence"] == "HIGH")
    medium_c = sum(1 for c in classified if c["confidence"] == "MEDIUM")
    draw.text((6, ly + 2), f"H={high_c}  M={medium_c}  {title}",
              fill=(255, 255, 100), font=font)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — scan all PDFs, aggregate statistics
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", action="store_true",
                        help="Print per-page detail table")
    args = parser.parse_args()

    show_pages  = args.pages
    all_results = {}
    global_high = global_med = global_rej = 0

    # Context-level accumulators for summary
    ctx_high_counts: dict[str, int] = defaultdict(int)

    print("=" * 76)
    print("  BRACE CLASSIFIER v2 — Universal Multi-Project Detector")
    print("=" * 76)

    for pdf_name, scale_ratio in SCAN_PDFS:
        path = os.path.join(UPLOAD_DIR, pdf_name)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {pdf_name}")
            continue

        ppf = scale_to_pts_per_foot(scale_ratio)
        doc = fitz.open(path)
        n   = len(doc)

        pdf_high = pdf_med = pdf_rej = 0
        pdf_pages_with_high = 0
        pdf_page_results    = {}
        overlay_pages       = []

        for pg in range(n):
            page = doc[pg]
            try:
                # Extract
                candidates       = extract_diagonals(page, ppf)
                # Context detection
                ctx, ctx_score   = classify_page_context(page)
                # Scale annotation detection
                scale_anns       = find_scale_annotations(page)
                # Classify with all layers
                classified       = classify(candidates, ctx, scale_anns, ppf)
                m                = page_metrics(classified, ctx)
                pdf_page_results[pg] = m
                pdf_high += m["high"]
                pdf_med  += m["medium"]
                pdf_rej  += m["reject"]
                if m["high"] > 0:
                    pdf_pages_with_high += 1
                    overlay_pages.append((pg, classified, ppf, page, ctx))
                    ctx_high_counts[ctx] += m["high"]
            except Exception:
                pass

        doc_summary = {
            "pages":           n,
            "pages_with_high": pdf_pages_with_high,
            "total_high":      pdf_high,
            "total_medium":    pdf_med,
            "total_reject":    pdf_rej,
            "page_detail":     pdf_page_results,
        }
        all_results[pdf_name] = doc_summary
        global_high += pdf_high
        global_med  += pdf_med
        global_rej  += pdf_rej

        short = pdf_name[:48]
        print(f"\n  {'─'*72}")
        print(f"  {short}")
        print(f"  scale=1:{scale_ratio:.0f}  pages={n}  "
              f"HIGH={pdf_high}  MED={pdf_med}  REJ={pdf_rej}  "
              f"p/HIGH={pdf_pages_with_high}")

        interesting = {pg: m for pg, m in pdf_page_results.items()
                       if m["high"] > 0 or m["medium"] > 0}

        if show_pages and interesting:
            print(f"\n    {'pg':>4}  {'H':>4}  {'M':>4}  {'R':>5}  "
                  f"{'context':<25}  {'reject reasons':<30}  high lengths (ft)")
            print(f"    {'─'*4}  {'─'*4}  {'─'*4}  {'─'*5}  {'─'*25}  "
                  f"{'─'*30}  {'─'*30}")
            for pg, m in sorted(interesting.items()):
                rr  = "  ".join(f"{k}:{v}" for k, v in m["reject_reasons"].items())
                hl  = " ".join(f"{x:.0f}" for x in m["high_lengths"][:8])
                ctx = m.get("context", "?")
                print(f"    {pg:>4}  {m['high']:>4}  {m['medium']:>4}  "
                      f"{m['reject']:>5}  {ctx:<25}  {rr:<30}  {hl}")
        elif interesting:
            pages_str = sorted(interesting.keys())
            print(f"    Pages with HIGH/MEDIUM: {pages_str}")
            top = sorted(interesting.items(), key=lambda x: x[1]["high"],
                         reverse=True)[:3]
            for pg, m in top:
                hl  = " ".join(f"{x:.0f}" for x in m["high_lengths"][:6])
                ctx = m.get("context", "?")
                print(f"    p{pg:02d}[{ctx[:20]}]: "
                      f"HIGH={m['high']} MED={m['medium']} lengths=[{hl}]")

        # Render overlays (first 4 pages with HIGH braces per PDF)
        pdf_slug = re.sub(r'[^A-Za-z0-9_]', '_', pdf_name[:30])
        rendered = 0
        for (pg, classified, ppf_, page, ctx) in overlay_pages:
            if rendered >= 4:
                break
            if sum(1 for c in classified if c["confidence"] == "HIGH") == 0:
                continue
            img = render_classified_overlay(page, classified, ppf_,
                                            f"p{pg}[{ctx[:12]}]")
            out = os.path.join(OUT_DIR, f"{pdf_slug}__p{pg:03d}.png")
            img.save(out)
            rendered += 1

        doc.close()

    # ── Global summary ────────────────────────────────────────────────────────
    total = global_high + global_med + global_rej or 1
    print(f"\n{'='*76}")
    print(f"  GLOBAL SUMMARY")
    print(f"{'='*76}")
    print(f"\n  Total candidates (all PDFs, all pages):")
    print(f"    HIGH   : {global_high:>6}  ({100*global_high/total:.1f}%)")
    print(f"    MEDIUM : {global_med:>6}  ({100*global_med/total:.1f}%)")
    print(f"    REJECT : {global_rej:>6}  ({100*global_rej/total:.1f}%)")
    print(f"    TOTAL  : {global_high+global_med+global_rej:>6}")

    print(f"\n  HIGH candidates by page context:")
    for ctx in ["braced_frame_elevation", "framing_plan", "roof_plan",
                "foundation_plan", "detail", "schedule_legend", "unknown"]:
        n = ctx_high_counts.get(ctx, 0)
        if n > 0:
            print(f"    {ctx:<30}: {n:>5}")

    print(f"\n  {'PDF':<48}  {'HIGH':>5}  {'MED':>5}  {'REJ':>6}  {'p/HIGH':>6}")
    print(f"  {'─'*48}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}")
    for pdf_name, _ in SCAN_PDFS:
        if pdf_name not in all_results:
            continue
        r = all_results[pdf_name]
        print(f"  {pdf_name[:48]:<48}  {r['total_high']:>5}  "
              f"{r['total_medium']:>5}  {r['total_reject']:>6}  "
              f"{r['pages_with_high']:>6}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    serializable = {}
    for pdf, data in all_results.items():
        serializable[pdf] = {
            "pages":           data["pages"],
            "pages_with_high": data["pages_with_high"],
            "total_high":      data["total_high"],
            "total_medium":    data["total_medium"],
            "total_reject":    data["total_reject"],
            "page_detail": {
                str(pg): {
                    "high":           m["high"],
                    "medium":         m["medium"],
                    "reject":         m["reject"],
                    "context":        m.get("context", "unknown"),
                    "high_lengths":   m["high_lengths"][:20],
                    "medium_lengths": m["medium_lengths"][:20],
                    "reject_reasons": m["reject_reasons"],
                }
                for pg, m in data["page_detail"].items()
            },
        }

    json_path = os.path.join(OUT_DIR, "brace_classifier_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  Overlays : {OUT_DIR}")
    print(f"  JSON     : {json_path}")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
