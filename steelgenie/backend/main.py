import fitz
import re
import math
import time
import base64
import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as PILImage, ImageEnhance, ImageFilter

# ── OpenCV-based raster beam line detector ───────────────────────────────────
try:
    from detection_cv import detect_beam_lines_raster as _detect_beam_lines_raster
    _RASTER_HOUGH_AVAILABLE = True
    print("[INIT] detection_cv.detect_beam_lines_raster loaded")
except Exception as _e:
    _RASTER_HOUGH_AVAILABLE = False
    print(f"[INIT] detection_cv not available — raster Hough disabled: {_e}")

# ── Load Environment ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_PATH = os.path.join(BASE_DIR, ".env")
    if os.path.exists(ENV_PATH):
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        print("[INIT] .env loaded")
    else:
        print("[INIT] .env not found")
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ──────────────────────────────────────────────────────────────────
supabase_client = None
try:
    from supabase import create_client as _sb_create
    _url = os.getenv("SUPABASE_URL")
    _key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if _url and _key:
        supabase_client = _sb_create(_url, _key)
        print("[INIT] Supabase ready")
except Exception as e:
    print(f"[INIT] Supabase error: {e}")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MEMBER_COLORS = {
    "beam":   "#EC4899",
    "column": "#3B82F6",
    "brace":  "#F59E0B",
}

STEEL_PATTERNS = [
    # W-sections: depth 1-2 digits, weight 2-3 digits (all real W shapes ≥ W4X13).
    # The (?!\d) lookahead prevents matching partial OCR reads like "W16X3" from
    # "W16X31" or "W16X3128" (garbage from adjacent load annotations).
    r'W\d{1,2}[Xx]\d{2,3}(?!\d)',
    # Abbreviated W-sections (depth only, no weight) — drawings that label beams
    # as "W12", "W16" etc. without lbs/ft.  The negative lookaheads/lookbehinds
    # prevent matching mid-word (e.g. "HSS10X8X3/8" or "W12X26").
    r'(?<![A-Z0-9])W\d{1,2}(?![Xx\d])',
    r'HSS[\d.]+[Xx][\d.]+(?:[Xx][\d.]+(?:/[\d.]+)?)?',  # HSS6X6, HSS6X6X1/4, HSS7.00X0.50
    r'L\d+[Xx]\d+',
    r'C\d+[Xx]\d+',
    r'MC\d+[Xx]\d+',
    r'ISA[\dXx]+',
    r'PIPE[\d.]+',
]

_GRID_LETTER = re.compile(r'^[A-Z](\.\d+)?$')
_GRID_NUMBER  = re.compile(r'^\d+(\.\d+)?$')

# Max distance (PDF points) between a profile label and a column symbol.
# Used for the one-to-one greedy symbol→profile matching in build_members.
SYMBOL_ASSOC_RADIUS = 65   # search radius for matching symbol to nearest label
SYMBOL_SNAP_RADIUS  = 110  # snap marker TO symbol position (wider — position only)

# Grid-intersection classification tolerance.
# Reduced from 35 → 20: beam labels that sit close to (but not at) a grid crossing
# were being falsely promoted to columns by TIER 2.  20 pt ≈ 0.28″ — tight enough
# to cover typical label offsets while excluding beams that frame INTO a column.
GRID_TOL = 20
# Grid-intersection SNAP radius (for marker placement — wider than classification)
GRID_SNAP_RADIUS = 50


# ── Scale conversion ──────────────────────────────────────────────────────────
def scale_to_pts_per_foot(scale_ratio: float) -> float:
    """
    Convert the frontend SCALE_OPTIONS ratio to PDF-points per foot.

    The frontend stores:  ratio = 12 / paper_inches_per_foot
    At 72 dpi:            pts_per_foot = 72 × paper_inches_per_foot
                                       = 72 × (12 / ratio)
                                       = 864 / ratio

    Examples
    --------
    1/8"=1'-0"  → ratio= 96 → pts_per_foot =  9.0
    3/16"=1'-0" → ratio= 64 → pts_per_foot = 13.5
    1/4"=1'-0"  → ratio= 48 → pts_per_foot = 18.0
    """
    if not scale_ratio or scale_ratio <= 0:
        return 0.0
    return 864.0 / scale_ratio


def compute_beam_span(cx: float, cy: float,
                      v_grid: list, h_grid: list,
                      pts_per_foot: float,
                      beam_dir: str = "H") -> dict | None:
    """
    Compute the beam span — both the length in feet AND the two physical
    endpoint positions (in raw PDF points) for rendering as a line overlay.

    Returns a dict:
        { "length_ft": float,
          "x1": float, "y1": float,   # start point (PDF pts)
          "x2": float, "y2": float }  # end point   (PDF pts)
    Returns None when the surrounding grid lines cannot be found.

    H-beam:  endpoints are (left_v_grid, cy) → (right_v_grid, cy)
    V-beam:  endpoints are (cx, top_h_grid)  → (cx, bottom_h_grid)
    """
    if beam_dir == "V":
        tops    = [gy for gy in h_grid if gy <= cy]
        bottoms = [gy for gy in h_grid if gy >  cy]
        if tops and bottoms:
            span_pt = min(bottoms) - max(tops)
            return {
                "length_ft": round(span_pt / pts_per_foot, 1) if pts_per_foot > 0 else 0.0,
                "x1": cx,         "y1": max(tops),
                "x2": cx,         "y2": min(bottoms),
            }
    else:  # "H" — default
        lefts  = [gx for gx in v_grid if gx <= cx]
        rights = [gx for gx in v_grid if gx >  cx]
        if lefts and rights:
            span_pt = min(rights) - max(lefts)
            return {
                "length_ft": round(span_pt / pts_per_foot, 1) if pts_per_foot > 0 else 0.0,
                "x1": max(lefts), "y1": cy,
                "x2": min(rights), "y2": cy,
            }

    return None


def _snap_to_grid_lines(x1: float, y1: float,
                        x2: float, y2: float,
                        bdir: str,
                        v_grid: list, h_grid: list,
                        snap_dist: float = 35.0) -> tuple:
    """
    Snap H/V beam endpoints to the nearest column grid line.

    Column grid lines (v_grid = vertical X positions, h_grid = horizontal Y
    positions) represent column CENTRELINES.  The beam line in the PDF often
    stops short of the centreline by half the column depth.  Snapping to the
    nearest grid line gives the true centre-to-centre structural span.

    snap_dist = 35 pt ≈ 3.9 ft at 1/8" scale — wide enough to bridge the
    half-column-depth gap PLUS any beam-face drawing offset (max ~3.5 ft
    combined), tight enough not to jump across to the wrong adjacent grid line
    (typical bay width >15 ft means the next column is always >150 pt away).

    Only EXTENDS the beam — never shortens it.
    """
    if bdir == "H" and v_grid:
        lx = min(x1, x2)   # left  endpoint X
        rx = max(x1, x2)   # right endpoint X

        # Left snap: find the nearest v_grid line to the LEFT of lx
        left_cands = [vx for vx in v_grid if lx - snap_dist <= vx < lx]
        if left_cands:
            new_lx = max(left_cands)   # closest grid left of lx
            if x1 <= x2:
                x1 = new_lx
            else:
                x2 = new_lx

        # Right snap: find the nearest v_grid line to the RIGHT of rx
        rx = max(x1, x2)   # recompute after possible left snap
        right_cands = [vx for vx in v_grid if rx < vx <= rx + snap_dist]
        if right_cands:
            new_rx = min(right_cands)
            if x1 <= x2:
                x2 = new_rx
            else:
                x1 = new_rx

    elif bdir == "V" and h_grid:
        ty = min(y1, y2)   # top    endpoint Y
        by_ = max(y1, y2)  # bottom endpoint Y

        # Top snap
        top_cands = [hy for hy in h_grid if ty - snap_dist <= hy < ty]
        if top_cands:
            new_ty = max(top_cands)
            if y1 <= y2:
                y1 = new_ty
            else:
                y2 = new_ty

        # Bottom snap
        by_ = max(y1, y2)
        bot_cands = [hy for hy in h_grid if by_ < hy <= by_ + snap_dist]
        if bot_cands:
            new_by = min(bot_cands)
            if y1 <= y2:
                y2 = new_by
            else:
                y1 = new_by

    return x1, y1, x2, y2, math.hypot(x2 - x1, y2 - y1)


def _snap_to_columns_along_axis(x1: float, y1: float,
                                x2: float, y2: float,
                                col_syms: list,
                                perp_tol: float = 30.0,
                                snap_dist: float = 50.0) -> tuple:
    """
    Snap both endpoints of a matched beam segment to the nearest column
    symbol that lies along the beam axis.

    In structural drawings the beam centreline is drawn from column FACE to
    column FACE (or even slightly short of the face).  The column I-symbol
    is placed at the column CENTRE.  This function bridges that gap so the
    extracted length equals the true centre-to-centre span.

    Parameters
    ----------
    perp_tol  : max perpendicular offset (pts) from beam axis to column centre
    snap_dist : max distance (pts) the endpoint can be FROM the column centre
                before we refuse to snap.  30 pt ≈ 2.7″ at ⅛″ scale — enough
                to cover half the depth of the deepest common column section
                while avoiding accidental jumps to an adjacent column symbol.
    """
    if not col_syms:
        return x1, y1, x2, y2, math.hypot(x2 - x1, y2 - y1)

    dx = x2 - x1
    dy = y2 - y1
    ln = math.hypot(dx, dy)
    if ln < 1.0:
        return x1, y1, x2, y2, ln

    ux, uy = dx / ln, dy / ln   # unit along beam
    px, py = -uy, ux            # unit perpendicular

    # Snap each end to the NEAREST column beyond it (not the furthest).
    # Taking the nearest is what makes a larger snap_dist safe: the endpoint
    # connects to the immediately-adjacent column and never jumps a bay.
    # Because targets are real column centres, this can never fly into empty
    # space the way a generic line-extension can.
    best_left_t  = None   # nearest column just BEFORE x1  (t < 0)
    best_right_t = None   # nearest column just AFTER  x2  (t > ln)

    for sym in col_syms:
        cx, cy = sym["cx"], sym["cy"]
        rv = (cx - x1, cy - y1)
        t    = rv[0] * ux + rv[1] * uy
        perp = abs(rv[0] * px + rv[1] * py)

        if perp > perp_tol:
            continue   # column is not on this beam's axis

        # Left end: column before x1 — keep the one closest to x1 (t nearest 0)
        if -snap_dist <= t < 0:
            if best_left_t is None or t > best_left_t:
                best_left_t = t

        # Right end: column past x2 — keep the one closest to x2 (t nearest ln)
        if ln < t <= ln + snap_dist:
            if best_right_t is None or t < best_right_t:
                best_right_t = t

    ox1, oy1 = x1, y1   # keep original origin for right-end computation
    if best_left_t is not None:
        x1 = ox1 + best_left_t * ux
        y1 = oy1 + best_left_t * uy
    if best_right_t is not None:
        x2 = ox1 + best_right_t * ux
        y2 = oy1 + best_right_t * uy

    return x1, y1, x2, y2, math.hypot(x2 - x1, y2 - y1)


def _extend_with_thin_segs(x1: float, y1: float, x2: float, y2: float,
                           thin_segs: list,
                           angle_tol_deg: float = 4.0,
                           gap_tol: float = 30.0,
                           colinear_tol: float = 5.0) -> tuple:
    """
    Extend a matched beam segment at both ends using thin continuation segments.

    In many CAD drawings the beam centreline transitions to a hairline stroke near
    the column connection zone.  The main matching step ignores those thin strokes
    but we save them in thin_segs.  After the best thick segment is found we stretch
    it to cover any collinear thin continuations, giving the accurate
    column-face-to-column-face beam length.

    Safety constraints (prevent over-extension):
      • angle must agree within angle_tol_deg (default 4°)
      • perpendicular offset must be < colinear_tol pts (default 5 pt ≈ 0.07″)
      • gap between segments must be < gap_tol pts (default 18 pt ≈ 0.25″)
      • at most 4 passes (handles chained thin segments, e.g. 2 short pieces at one end)

    Returns (x1, y1, x2, y2, length).
    """
    if not thin_segs:
        return x1, y1, x2, y2, math.hypot(x2 - x1, y2 - y1)

    for _pass in range(4):
        dx = x2 - x1
        dy = y2 - y1
        ln = math.hypot(dx, dy)
        if ln < 1.0:
            break
        ux, uy = dx / ln, dy / ln
        px, py = -uy, ux
        ang = math.atan2(dy, dx)
        if ang < 0:
            ang += math.pi

        ext_left  = 0.0   # most-negative t on the left side (0 = no extension)
        ext_right = ln    # largest t on the right side (ln = no extension)

        for (sx1, sy1, sx2, sy2, _) in thin_segs:
            sdx, sdy = sx2 - sx1, sy2 - sy1
            sang = math.atan2(sdy, sdx)
            if sang < 0:
                sang += math.pi
            da = abs(ang - sang)
            if da > math.pi / 2:
                da = math.pi - da
            if math.degrees(da) > angle_tol_deg:
                continue

            # Perpendicular distance — both endpoints checked, take the minimum
            rv1x, rv1y = sx1 - x1, sy1 - y1
            rv2x, rv2y = sx2 - x1, sy2 - y1
            pd = min(abs(rv1x * px + rv1y * py), abs(rv2x * px + rv2y * py))
            if pd > colinear_tol:
                continue

            # Parametric projections onto our axis
            t1 = rv1x * ux + rv1y * uy
            t2 = rv2x * ux + rv2y * uy
            if t1 > t2:
                t1, t2 = t2, t1

            # Left end: thin seg reaches before our current x1 but is close to it
            if t1 < 0 and t2 >= -gap_tol:
                ext_left = min(ext_left, t1)

            # Right end: thin seg reaches past our current x2 but is close to it
            if t2 > ln and t1 <= ln + gap_tol:
                ext_right = max(ext_right, t2)

        changed = False
        ox1, oy1 = x1, y1   # keep original origin for right-end computation
        if ext_left < 0:
            x1 = ox1 + ext_left * ux
            y1 = oy1 + ext_left * uy
            changed = True
        if ext_right > ln:
            x2 = ox1 + ext_right * ux   # always relative to original origin
            y2 = oy1 + ext_right * uy
            changed = True
        if not changed:
            break

    return x1, y1, x2, y2, math.hypot(x2 - x1, y2 - y1)


def detect_beam_lines(page, profiles: list, plan_bounds: tuple,
                      pts_per_foot: float = 0.0,
                      column_symbols: list = None,
                      v_grid: list = None,
                      h_grid: list = None) -> dict:
    """
    PRIMARY beam detection: find structural centerlines in the PDF vector drawing
    and match them to steel-section text labels.

    In CAD-exported structural framing plans every beam IS drawn as a line segment
    on its centreline.  The profile label (e.g. "W24X76") sits right on — or very
    close to — that line at mid-span.

    Strategy
    --------
    1. Collect ALL line segments inside the plan boundary that are 30–650 pt long
       (any angle — H, V, or diagonal).  Diagonal beams appear in irregular
       framing plans (e.g. Area B skewed grids) and must not be excluded.
    2. For each profile label (cx, cy) find the closest line using perpendicular
       distance: project the label onto the infinite extension of each line and
       measure the distance to the closest point on the segment (±15 % extension
       so labels near span ends still match).  Accept if distance < LABEL_R.
    3. Return a dict  { profile_idx: {"x1","y1","x2","y2","dir","length_pt"} }
       for every profile that was successfully matched to a drawn line.
       dir is "H" / "V" / "D" (diagonal).
       Unmatched profiles fall back to the grid-based span calculation.

    This gives us EXACT endpoints and EXACT length from the drawing geometry —
    no grid guessing needed.
    """
    bx0, by0, bx1, by1 = plan_bounds
    # MIN_LEN = 45 pt ≈ 5 ft at 1/8" scale.
    # This filters two things:
    #  1. Real ticks / arrowheads / hatch lines (always <30 pt)
    #  2. Exploded dash segments — many CAD exporters draw dashed lines as
    #     many short SOLID segments with gaps rather than a single path with
    #     a PDF dash pattern.  Typical structural drawing dash lengths are
    #     3–30 pt, so 45 pt cleanly separates them from beam centrelines.
    # Dynamically scale MIN_LEN to allow short beams (down to 3 ft) at any scale,
    # but never drop below 15 pt to keep filtering out small hatch marks.
    MIN_LEN  = max(12, int(pts_per_foot * 1.0)) if pts_per_foot > 0 else 15
    # Cap at 80 ft using the drawing scale — prevents full-plan dimension/
    # annotation lines from being matched as beam centrelines.
    # Fall back to 700 pt (≈78 ft at 1/8") when scale is unknown.
    MAX_LEN  = int(pts_per_foot * 80) if pts_per_foot > 0 else 700
    # 180 pt: raised from 130 so labels pushed far from their beam in congested
    # or large-scale drawings (3/16", 1/4") still match.  Proximity scoring still
    # picks the closest line — a larger radius does NOT increase false matches.
    LABEL_R  = 180

    # ── Per-direction max-length guards ──────────────────────────────────────
    # MAX_H_MATCH / MAX_V_MATCH cap horizontal / vertical lines separately.
    # Primary cap = 1.5× widest structural bay — prevents full-width grid and
    # boundary lines from being matched as beam centrelines.
    # Fallback cap = 65% of plan dimension.
    # Diagonal lines use their own cap: hypot(MAX_H, MAX_V).
    def _max_bay(grid):
        sg = sorted(grid)
        if len(sg) < 2:
            return float("inf")
        return max(b - a for a, b in zip(sg, sg[1:]))
    _mb_w = _max_bay(v_grid or [])
    _mb_h = _max_bay(h_grid or [])
    
    # Floor cap: if grid detection finds only a few bubbles, _max_bay is tiny and
    # MAX_H/V_MATCH collapses — real long beams get rejected as "too long".
    # 40 ft is the safe floor for typical buildings; the 1.5× bay multiplier
    # automatically scales up for large hospitals / warehouses when grid IS detected.
    _MIN_CAP = int(pts_per_foot * 40) if pts_per_foot > 0 else 400
    MAX_H_MATCH = max(_mb_w * 1.5, _MIN_CAP) if _mb_w < float("inf") else MAX_LEN
    MAX_V_MATCH = max(_mb_h * 1.5, _MIN_CAP) if _mb_h < float("inf") else MAX_LEN
    if plan_bounds:
        _pb_w = plan_bounds[2] - plan_bounds[0]
        _pb_h = plan_bounds[3] - plan_bounds[1]
        MAX_H_MATCH = min(MAX_H_MATCH, _pb_w * 0.95)   # raised from 0.90 — edge beams
        MAX_V_MATCH = min(MAX_V_MATCH, _pb_h * 0.95)

    all_lines: list[tuple] = []   # (x1, y1, x2, y2, length)
    thin_segs: list[tuple] = []   # thin-stroke segments at beam/column junctions

    try:
        for d in page.get_drawings():
            # ── Skip dashed / dotted paths ────────────────────────────────────
            # Type A — PDF dash pattern: check that the bracket content of
            #   d["dashes"] is empty.  A solid line is "" or "[] 0" (empty
            #   bracket array).  Any content inside the brackets, e.g.
            #   "[3 3] 0", "[0.5 1.5] 0", means dashed/dotted → skip.
            #   Using regex on bracket content is more robust than exact-
            #   string comparison, which would miss variants like "[] 0.0".
            _dashes = (d.get("dashes") or "").strip()
            _da = re.match(r'\[([^\]]*)\]', _dashes)
            if _da and _da.group(1).strip():
                continue   # non-empty bracket content → dashed line

            # Type B — Exploded dashes: CAD draws each dash as a tiny SOLID
            #   segment with a gap to the next.  No dash property is set.
            #   Handled by MIN_LEN=45 (rejects segments shorter than ~5 ft).

            # Type C — Thin construction / hidden lines that carry no dash
            #   property but are drawn hairline-thin in CAD (lineweight ≤ 0.3 pt).
            #   Structural beam centrelines always have a measurable weight.
            #   Skip any path with an explicitly set width < 0.3 pt for the main
            #   matching pool — but save valid segments into thin_segs so we can
            #   extend a matched thick segment to its true column-face length.
            #   (Width = 0 or None means "default" in some exporters — keep those.)
            _lw = d.get("width") or 0
            if 0 < _lw < 0.3:
                for _item in d.get("items", []):
                    if _item[0] != "l":
                        continue
                    try:
                        _p1, _p2 = _item[1], _item[2]
                        _tln = math.hypot(abs(_p2.x - _p1.x), abs(_p2.y - _p1.y))
                        # Use a much smaller minimum than all_lines (MIN_LEN=45).
                        # Column-zone beam stubs can be very short (a few pts) but
                        # are still valid continuations.  The collinearity check in
                        # _extend_with_thin_segs keeps spurious tiny marks out.
                        if _tln < 3.0 or _tln > MAX_LEN:
                            continue
                        _tmx = (_p1.x + _p2.x) / 2
                        _tmy = (_p1.y + _p2.y) / 2
                        # 150 pt midpoint tolerance matches all_lines — catches
                        # thin column-zone stubs for edge/cantilever beams.
                        _TMID = 150
                        if not (bx0 - _TMID <= _tmx <= bx1 + _TMID and
                                by0 - _TMID <= _tmy <= by1 + _TMID):
                            continue
                        # Both endpoints must stay within plan bounds (+ tolerance).
                        # Raised to 250 pt to match all_lines _EP_TOL (wings of building).
                        _TEP = 250
                        if (min(_p1.x, _p2.x) < bx0 - _TEP or
                                max(_p1.x, _p2.x) > bx1 + _TEP):
                            continue
                        if (min(_p1.y, _p2.y) < by0 - _TEP or
                                max(_p1.y, _p2.y) > by1 + _TEP):
                            continue
                        thin_segs.append((_p1.x, _p1.y, _p2.x, _p2.y, _tln))
                    except Exception:
                        continue
                continue

            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                try:
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    ln = math.hypot(dx, dy)
                    if ln < MIN_LEN or ln > MAX_LEN:
                        continue
                    # No plan_bounds filter on lines — collect from the entire page.
                    # plan_bounds is unreliable on large multi-bay drawings (the
                    # detected boundary covers only the grid-bubble region, which may
                    # be 30–50% of the actual framing extent on Area-B / hospital sheets).
                    # Label→line matching (LABEL_R) is the proximity guard; the
                    # structural length filter (MIN_LEN / MAX_LEN) removes arrows,
                    # tick marks, and title-block rules.  Cross-drawing false matches
                    # don't occur because a label in one drawing area won't be within
                    # LABEL_R of a line in a distant drawing area.
                    all_lines.append((p1.x, p1.y, p2.x, p2.y, ln))
                except Exception:
                    continue
    except Exception:
        pass

    h_c = sum(1 for (x1,y1,x2,y2,ln) in all_lines if abs(x2-x1) > abs(y2-y1)*2)
    v_c = sum(1 for (x1,y1,x2,y2,ln) in all_lines if abs(y2-y1) > abs(x2-x1)*2)
    print(f"[BEAM_LINES] H:{h_c}  V:{v_c}  Diagonal:{len(all_lines)-h_c-v_c}  "
          f"Total:{len(all_lines)}")

    result: dict[int, dict] = {}

    # Column-centreline X / Y positions for the endpoint snap (Pass 3d).
    # Both grid lines AND detected column symbols mark column centres, so a beam
    # end can land true centre-to-centre whether the column shows up as a grid
    # line or only as an I/H symbol.
    _col_snap_x = sorted(set([round(x, 1) for x in (v_grid or [])] +
                             [round(s["cx"], 1) for s in (column_symbols or [])]))
    _col_snap_y = sorted(set([round(y, 1) for y in (h_grid or [])] +
                             [round(s["cy"], 1) for s in (column_symbols or [])]))

    # ── Label → line matcher (one-to-one) ────────────────────────────────────
    # _find_best_line scores every candidate line for a label and returns the
    # best (line, score, midpoint-key).  `claimed` is a set of midpoint-keys of
    # lines already taken by a closer label, so each drawn beam line is matched
    # by AT MOST ONE label.  Without this, dense/closely-spaced parallel beams
    # (e.g. canopy framing) all grab the same nearest line, orphaning the rest
    # and leaving real beams with no marker.
    def _mid_key(lx1, ly1, lx2, ly2):
        # ~6 pt buckets — two DISTINCT beams are never <0.7 ft apart, so this
        # only ever collides for the same physical line.
        return (round((lx1 + lx2) / 2 / 6.0), round((ly1 + ly2) / 2 / 6.0))

    def _find_best_line(pcx, pcy, label_is_h, label_is_v,
                        enforce_dir, line_pool=None, claimed=None):
        _best = None
        _best_score = -1.0
        _best_mid = None
        for (lx1, ly1, lx2, ly2, ln) in (line_pool if line_pool is not None else all_lines):
            if claimed is not None:
                _mk = _mid_key(lx1, ly1, lx2, ly2)
                if _mk in claimed:
                    continue
            ldx = lx2 - lx1
            ldy = ly2 - ly1
            adx = abs(ldx)
            ady = abs(ldy)

            # Direction guard — label orientation must agree with line direction.
            if enforce_dir:
                if label_is_h and (ady > adx * 2):
                    continue
                if label_is_v and (adx > ady * 2):
                    continue

            # Per-direction max-length guard — reject grid/boundary lines
            if ady > adx * 2 and ln > MAX_V_MATCH:        # vertical line
                continue
            if adx > ady * 2 and ln > MAX_H_MATCH:        # horizontal line
                continue
            if not (adx > ady * 2) and not (ady > adx * 2):  # diagonal
                _diag_w = _mb_w if _mb_w < float("inf") else MAX_H_MATCH
                _diag_h = _mb_h if _mb_h < float("inf") else MAX_V_MATCH
                _max_diag = min(math.hypot(_diag_w, _diag_h) * 3.5,
                                math.hypot(_pb_w, _pb_h) * 0.85 if plan_bounds else MAX_LEN)
                if ln > _max_diag:
                    continue

            # Parametric projection onto line segment
            t = ((pcx - lx1) * ldx + (pcy - ly1) * ldy) / (ln * ln)
            # Allow ±45 % extension: raised from ±30 % so labels placed beyond
            # beam ends (common in congested areas and at cantilever tips) still
            # match their beam.  The proximity score naturally deprioritises
            # far-end labels when a closer line exists.
            if t < -0.45 or t > 1.45:
                continue
            t_c = max(0.0, min(1.0, t))
            px_proj = lx1 + t_c * ldx
            py_proj = ly1 + t_c * ldy
            dist = math.hypot(pcx - px_proj, pcy - py_proj)
            if dist >= LABEL_R:
                continue

            # Score = proximity × (0.97 + 0.03 × t_center)
            proximity  = 1.0 - dist / LABEL_R
            t_center   = 1.0 - 2.0 * abs(t_c - 0.5)
            score = proximity * (0.97 + 0.03 * t_center)
            if score > _best_score:
                _best_score = score
                _best       = (lx1, ly1, lx2, ly2, ln)
                _best_mid   = _mid_key(lx1, ly1, lx2, ly2)
        return _best, _best_score, _best_mid

    def _match_profile(p, claimed):
        """Run all match passes for one profile against unclaimed lines."""
        pcx, pcy = p["cx"], p["cy"]
        lbw = p.get("bbox_w", 20.0)
        lbh = p.get("bbox_h",  8.0)
        label_is_h = lbw > lbh * 3.5
        label_is_v = lbh > lbw * 3.5
        b, s, m = _find_best_line(pcx, pcy, label_is_h, label_is_v, True, None, claimed)
        if not b:
            b, s, m = _find_best_line(pcx, pcy, label_is_h, label_is_v, False, None, claimed)
        if not b and thin_segs:
            _thin_long = [(x1, y1, x2, y2, ln) for (x1, y1, x2, y2, ln) in thin_segs
                          if ln >= MIN_LEN]
            if _thin_long:
                b, s, m = _find_best_line(pcx, pcy, label_is_h, label_is_v, True, _thin_long, claimed)
                if not b:
                    b, s, m = _find_best_line(pcx, pcy, label_is_h, label_is_v, False, _thin_long, claimed)
        return b, s, m

    # Pre-pass: each profile's independent best score → process closest-first so
    # the strongest (label-on-its-own-line) matches claim their line before
    # weaker ones, distributing labels across all lines instead of clustering.
    _prelim = []
    for _pi in range(len(profiles)):
        _, _sc, _ = _match_profile(profiles[_pi], None)
        _prelim.append((_sc, _pi))
    _order = [pi for _sc, pi in sorted(_prelim, key=lambda z: -z[0])]

    _claimed_lines: set = set()
    for p_idx in _order:
        p = profiles[p_idx]
        pcx, pcy = p["cx"], p["cy"]
        best, _bscore, _bmid = _match_profile(p, _claimed_lines)
        if best:
            if _bmid is not None:
                _claimed_lines.add(_bmid)
            lx1, ly1, lx2, ly2, ln = best
            # ── Pass 0: collinear thick-segment chaining (COLUMN-GATED) ───────
            # Beams are frequently drawn as MULTIPLE short collinear segments —
            # CAD exporters break the centreline at every girder crossing.  The
            # label then matches only ONE short piece, so the rendered line is a
            # tiny stub instead of the full span (the bug being fixed here).
            #
            # Recover the full length by chaining every REAL drawn segment that
            # is collinear with the matched one (same direction within 4°, same
            # infinite line within ~5 pt).  This follows only ACTUAL drawn
            # geometry, so it can never extend into empty space.
            #
            # CRITICAL GUARD — column gating: two DIFFERENT beams that meet at a
            # column are also collinear, so naive chaining bridges them into one
            # cross-bay line (the over-extension regression).  A real single beam
            # is only ever broken at GIRDER crossings (mid-span), never at a
            # column.  So after chaining we truncate any extension that ran PAST
            # a column back TO that column: a column between the matched piece
            # and the chained end means we crossed into a neighbouring beam.
            _om_x1, _om_y1, _om_x2, _om_y2 = lx1, ly1, lx2, ly2   # original extent
            _c1, _c2, _c3, _c4, _cln = _extend_with_thin_segs(
                lx1, ly1, lx2, ly2, all_lines,
                gap_tol=max(15.0, pts_per_foot * 1.0))
            _cadx, _cady = abs(_c3 - _c1), abs(_c4 - _c2)
            _is_h_chain = _cadx > _cady
            _ccap = MAX_H_MATCH if _is_h_chain else MAX_V_MATCH
            if _cln <= _ccap and (_cln > ln + 1):
                # Gate on REAL column symbols only — NOT grid lines.  A grid line
                # crossing does not mean a column exists at THIS beam's position
                # (a vertical infill beam crosses many row grid lines but frames
                # girder-to-girder with no column between).  Truncate only where
                # an actual detected column sits ON the beam axis between the
                # matched piece and the chained end — that is a real beam-to-beam
                # junction (two members meeting at a column), not one beam.
                _PERP = 22.0   # column centre must lie within this of the axis
                if _is_h_chain:
                    _yl = (_c2 + _c4) / 2.0
                    _ol, _orr = min(_om_x1, _om_x2), max(_om_x1, _om_x2)
                    _nl, _nr  = min(_c1, _c3),       max(_c1, _c3)
                    _colx = [s["cx"] for s in (column_symbols or [])
                             if abs(s["cy"] - _yl) < _PERP]
                    _lc = [c for c in _colx if _nl < c < _ol - 2]
                    if _lc:
                        _nl = max(_lc)         # stop at column nearest matched piece
                    _rc = [c for c in _colx if _orr + 2 < c < _nr]
                    if _rc:
                        _nr = min(_rc)
                    lx1, ly1, lx2, ly2 = _nl, _yl, _nr, _yl
                    ln = abs(_nr - _nl)
                else:
                    _xl = (_c1 + _c3) / 2.0
                    _ot, _ob = min(_om_y1, _om_y2), max(_om_y1, _om_y2)
                    _nt, _nb = min(_c2, _c4),       max(_c2, _c4)
                    _coly = [s["cy"] for s in (column_symbols or [])
                             if abs(s["cx"] - _xl) < _PERP]
                    _tc = [c for c in _coly if _nt < c < _ot - 2]
                    if _tc:
                        _nt = max(_tc)
                    _bc = [c for c in _coly if _ob + 2 < c < _nb]
                    if _bc:
                        _nb = min(_bc)
                    lx1, ly1, lx2, ly2 = _xl, _nt, _xl, _nb
                    ln = abs(_nb - _nt)
            # Remember the matched centreline extent (AFTER collinear chaining)
            # BEFORE the snap passes.  The final clamp below bounds the COMBINED
            # growth of the snap passes (column snap, intersection snap) to
            # _EXT_MAX past this extent, so a beam can never be inflated far past
            # its real drawn line — this stops hairline grid/border segments from
            # being chained into huge cross-the-sheet "beams" on dense drawings.
            _draw_x1, _draw_y1 = lx1, ly1
            _draw_x2, _draw_y2 = lx2, ly2
            # ── Pass 1: hairline stubs at column connection zones ─────────────
            # The beam centreline transitions to a thin stroke (<0.3 pt) inside
            # the column zone.  Extend using those saved thin_segs.
            if thin_segs:
                lx1, ly1, lx2, ly2, ln = _extend_with_thin_segs(
                    lx1, ly1, lx2, ly2, thin_segs)
            # ── Pass 3a: column-symbol snap ───────────────────────────────────
            # Snap endpoints to the nearest real column lying along the beam
            # axis.  Targets = detected I/H column symbols PLUS grid-line
            # intersections (also exact column centres).  Both are guaranteed
            # column positions, so this can connect a short-drawn beam to its
            # column without any risk of flying into empty space.
            _col_targets = list(column_symbols or [])
            if v_grid and h_grid:
                _col_targets += [{"cx": gx, "cy": gy}
                                 for gx in v_grid for gy in h_grid]
            if _col_targets:
                lx1, ly1, lx2, ly2, ln = _snap_to_columns_along_axis(
                    lx1, ly1, lx2, ly2, _col_targets)
            # ── Pass 3b: physical intersection snap ───────────────────────────────
            # Geometrically extend each beam endpoint by up to _EXT_MAX points
            # to the nearest crossing line, so beams meet at the column/beam
            # centreline rather than stopping at the column face.
            #
            # Works for H, V AND diagonal (D) beams — critical for rotated-grid
            # drawings where all structural beams are at 30-45° angles.
            #
            # Algorithm: for each endpoint, project the beam direction forward/
            # backward by up to _EXT_MAX, find the nearest line segment that
            # actually intersects that extension, and snap to the intersection.
            # EXT_MAX: how far each endpoint can be extended to reach a column/beam.
            # Scale with drawing — gap from beam-end to column CL is typically
            # 6–12 inches (half column flange) + any drafter shortfall, ≈ 2–8 ft.
            # Formula: 8 ft × pts_per_foot; clamp 40–150 pt for unknown scales.
            # 15 ft allows reaching the column CL even when only a short stub
            # of the beam is drawn (some drafters draw to column face, or split
            # beams into bay-by-bay segments).  Nearest-snap stops at the first
            # real crossing line so there is no overrun.
            # Beam-face → column-centreline gap is only half a column depth
            # (< 1 ft).  Allow 2.5 ft of extension to bridge that plus minor
            # drafter slack.  Crucially, when NO column line is within this
            # range the endpoint KEEPS its drawn position instead of flying out
            # to a far grid/perimeter line in empty space (the overshoot bug).
            _EXT_MAX = max(15, min(45, pts_per_foot * 2.5)) if pts_per_foot > 0 else 25.0
            _adx, _ady = abs(lx2 - lx1), abs(ly2 - ly1)
            _is_H = _adx > _ady * 2
            _is_V = _ady > _adx * 2

            # ── General line-segment intersection helper ──────────────────────
            def _seg_intersect_t(ax1, ay1, adx, ady, bx1, by1, bx2, by2):
                """
                Return t along beam ray (ax1+t*adx, ay1+t*ady) where it
                intersects segment B, or None if no intersection.
                t < 0  = behind endpoint, 0 = at endpoint, t > 0 = ahead.
                Only returns when the intersection is within segment B (u ∈ [0,1]).
                """
                bdx = bx2 - bx1
                bdy = by2 - by1
                denom = adx * bdy - ady * bdx
                if abs(denom) < 1e-9:
                    return None
                t = ((bx1 - ax1) * bdy - (by1 - ay1) * bdx) / denom
                u = ((bx1 - ax1) * ady - (by1 - ay1) * adx) / denom
                if -0.05 <= u <= 1.05:
                    return t
                return None

            # ── Pre-filter candidate lines ────────────────────────────────────
            # Only lines long enough to be a real structural element (column
            # web/flange or girder) qualify as snap targets.  Short ticks,
            # dimension arrows, and hatch marks are excluded so the nearest-snap
            # doesn't stop at a 5 pt annotation and leave the beam 60 pt short
            # of the actual column.
            # Minimum: 5 ft equivalent at the drawing scale, floor 30 pt.
            # 5 ft filters out dimension ticks, hatch lines, and connection
            # detail marks while keeping all real column and girder lines.
            _SNAP_MIN = max(30, pts_per_foot * 5) if pts_per_foot > 0 else 30

            if _is_H:
                _cand_lines = [(px1, py1, px2, py2)
                               for (px1, py1, px2, py2, pln) in all_lines
                               if abs(py2 - py1) > abs(px2 - px1) * 2
                               and pln >= _SNAP_MIN]
                # Also include thin crossing lines (hairline column/girder strokes)
                _cand_lines += [(px1, py1, px2, py2)
                                for (px1, py1, px2, py2, pln) in thin_segs
                                if abs(py2 - py1) > abs(px2 - px1) * 2
                                and pln >= _SNAP_MIN]
            elif _is_V:
                _cand_lines = [(px1, py1, px2, py2)
                               for (px1, py1, px2, py2, pln) in all_lines
                               if abs(px2 - px1) > abs(py2 - py1) * 2
                               and pln >= _SNAP_MIN]
                # Also include thin crossing lines (hairline column/girder strokes)
                _cand_lines += [(px1, py1, px2, py2)
                                for (px1, py1, px2, py2, pln) in thin_segs
                                if abs(px2 - px1) > abs(py2 - py1) * 2
                                and pln >= _SNAP_MIN]
            else:
                # Diagonal beam — collect lines NOT parallel to it
                # (angle difference > 20°  ≈  dot product < cos20 = 0.94)
                _blen = math.hypot(lx2 - lx1, ly2 - ly1)
                _bnx  = (lx2 - lx1) / _blen if _blen > 0 else 1.0
                _bny  = (ly2 - ly1) / _blen if _blen > 0 else 0.0
                _cand_lines = []
                for (px1, py1, px2, py2, pln) in all_lines + thin_segs:
                    if pln < _SNAP_MIN:
                        continue
                    _pnx = (px2 - px1) / pln
                    _pny = (py2 - py1) / pln
                    _dot = abs(_bnx * _pnx + _bny * _pny)
                    if _dot < 0.94:  # not parallel
                        _cand_lines.append((px1, py1, px2, py2))

            # ── Extend endpoint A (the "start" end of the beam) ──────────────
            # Direction from A outward = -(lx2-lx1, ly2-ly1) direction
            _blen = math.hypot(lx2 - lx1, ly2 - ly1) or 1.0
            _fwdx = (lx2 - lx1) / _blen   # forward unit vector
            _fwdy = (ly2 - ly1) / _blen
            _best_t_A = float("inf")   # take NEAREST crossing line, not furthest
            _snap_A = None
            for (px1, py1, px2, py2) in _cand_lines:
                # Shoot ray backward from lx1,ly1
                t = _seg_intersect_t(lx1, ly1, -_fwdx, -_fwdy,
                                     px1, py1, px2, py2)
                if t is not None and 0.0 < t <= _EXT_MAX:
                    if t < _best_t_A:
                        _best_t_A = t
                        _snap_A = (lx1 - _fwdx * t, ly1 - _fwdy * t)
            if _snap_A:
                lx1, ly1 = _snap_A

            # ── Extend endpoint B (the "end" end of the beam) ────────────────
            _best_t_B = float("inf")   # take NEAREST crossing line, not furthest
            _snap_B = None
            for (px1, py1, px2, py2) in _cand_lines:
                # Shoot ray forward from lx2,ly2
                t = _seg_intersect_t(lx2, ly2, _fwdx, _fwdy,
                                     px1, py1, px2, py2)
                if t is not None and 0.0 < t <= _EXT_MAX:
                    if t < _best_t_B:
                        _best_t_B = t
                        _snap_B = (lx2 + _fwdx * t, ly2 + _fwdy * t)
            if _snap_B:
                lx2, ly2 = _snap_B

            # ── Pass 3d: column-centreline snap ───────────────────────────────
            # Grid lines ARE the column centrelines.  A drawn beam stops at the
            # column FACE — ~1–2 ft short of the centre.  Snap each endpoint to
            # the NEAREST grid line, but only within _EXT_MAX (so it reaches the
            # adjacent column centre and never jumps to the next bay, and never
            # overshoots — nearest-snap pulls an over-long end back too).
            # This is what makes beams land true centre-to-centre, generically.
            if _is_H and _col_snap_x:
                _gx1 = min(_col_snap_x, key=lambda gx: abs(lx1 - gx))
                _gx2 = min(_col_snap_x, key=lambda gx: abs(lx2 - gx))
                # Only snap when the two ends reach DIFFERENT columns — never
                # collapse both ends onto one column (which would erase the
                # beam) and never lose the span of a short in-bay beam.
                if abs(_gx1 - _gx2) > 5:
                    if abs(lx1 - _gx1) <= _EXT_MAX:
                        lx1 = _gx1
                    if abs(lx2 - _gx2) <= _EXT_MAX:
                        lx2 = _gx2
            elif _is_V and _col_snap_y:
                _gy1 = min(_col_snap_y, key=lambda gy: abs(ly1 - gy))
                _gy2 = min(_col_snap_y, key=lambda gy: abs(ly2 - gy))
                if abs(_gy1 - _gy2) > 5:
                    if abs(ly1 - _gy1) <= _EXT_MAX:
                        ly1 = _gy1
                    if abs(ly2 - _gy2) <= _EXT_MAX:
                        ly2 = _gy2

            # ── FINAL ANTI-OVERSHOOT CLAMP ────────────────────────────────────
            # No endpoint may sit more than _EXT_MAX beyond the true drawn extent
            # captured above.  This bounds the COMBINED effect of every snap pass
            # (column-symbol snap, intersection snap), so a beam can never fly
            # past its drawn line into empty space — on any drawing, generically.
            # Measured along the original drawn axis; movement back toward the
            # beam (shrinking) is never clamped.
            _odx = _draw_x2 - _draw_x1
            _ody = _draw_y2 - _draw_y1
            _oln = math.hypot(_odx, _ody)
            if _oln > 1.0:
                _oux, _ouy = _odx / _oln, _ody / _oln
                # Endpoint A: outward direction is -axis; clamp extension to _EXT_MAX
                _extA = -((lx1 - _draw_x1) * _oux + (ly1 - _draw_y1) * _ouy)
                if _extA > _EXT_MAX:
                    lx1 = _draw_x1 - _oux * _EXT_MAX
                    ly1 = _draw_y1 - _ouy * _EXT_MAX
                # Endpoint B: outward direction is +axis
                _extB = (lx2 - _draw_x2) * _oux + (ly2 - _draw_y2) * _ouy
                if _extB > _EXT_MAX:
                    lx2 = _draw_x2 + _oux * _EXT_MAX
                    ly2 = _draw_y2 + _ouy * _EXT_MAX

            ln = math.hypot(lx2 - lx1, ly2 - ly1)
            adx = abs(lx2 - lx1)
            ady = abs(ly2 - ly1)
            if adx > ady * 2:
                bdir = "H"
            elif ady > adx * 2:
                bdir = "V"
            else:
                bdir = "D"   # diagonal beam
            result[p_idx] = {
                "x1": lx1, "y1": ly1,
                "x2": lx2, "y2": ly2,
                "dir":       bdir,
                "length_pt": ln,
            }

    matched = len(result)
    print(f"[BEAM_LINES] {matched}/{len(profiles)} profiles matched "
          f"to drawn beam lines")
    return result, all_lines


def detect_beam_directions(page, profiles: list, plan_bounds: tuple) -> dict:
    """
    Determine whether each beam profile label lies on a horizontal or vertical
    structural member by finding the nearest significant vector line segment.

    Strategy
    --------
    • Collect all line segments from page vector drawings that are:
        – Inside the plan boundary
        – Between MIN_LEN and MAX_LEN pts long (ignores ticks and grid lines)
        – Clearly horizontal (dx/dy > 2) or vertical (dy/dx > 2)
    • For each profile, whichever orientation class has a member closer than
      SEARCH_R pts wins.  Ties (or nothing within range) default to "H".

    Returns
    -------
    dict  profile_idx → "H" | "V"
    """
    bx0, by0, bx1, by1 = plan_bounds

    MIN_LEN  = 40    # ignore ticks, hatching, and annotation lines
    MAX_LEN  = 600   # ignore full-width grid lines and sheet border
    DIR_RATIO = 2.0  # dx/dy (or dy/dx) must exceed this to be "clearly" H or V
    SEARCH_R  = 60   # max distance from label to candidate line midpoint

    h_lines: list[tuple[float, float]] = []
    v_lines: list[tuple[float, float]] = []

    try:
        for d in page.get_drawings():
            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                try:
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    ln = math.hypot(dx, dy)
                    if ln < MIN_LEN or ln > MAX_LEN:
                        continue
                    lx = (p1.x + p2.x) / 2
                    ly = (p1.y + p2.y) / 2
                    if not (bx0 <= lx <= bx1 and by0 <= ly <= by1):
                        continue
                    if dx > dy * DIR_RATIO:
                        h_lines.append((lx, ly))
                    elif dy > dx * DIR_RATIO:
                        v_lines.append((lx, ly))
                except Exception:
                    continue
    except Exception:
        pass

    print(f"[BEAM_DIR] H-lines: {len(h_lines)}  V-lines: {len(v_lines)}")

    directions: dict[int, str] = {}
    for p_idx, p in enumerate(profiles):
        pcx, pcy = p["cx"], p["cy"]
        d_h = min((math.hypot(pcx - lx, pcy - ly) for lx, ly in h_lines),
                  default=float("inf"))
        d_v = min((math.hypot(pcx - lx, pcy - ly) for lx, ly in v_lines),
                  default=float("inf"))

        if d_h <= SEARCH_R or d_v <= SEARCH_R:
            directions[p_idx] = "H" if d_h <= d_v else "V"
        else:
            directions[p_idx] = "H"  # default — horizontal beam

    return directions


# ── Profile helpers ───────────────────────────────────────────────────────────
def normalize_profile(text: str) -> str:
    t = text.upper().strip()
    t = re.sub(r'[^A-Z0-9X/]', '', t)
    t = re.sub(r'^VV', "W", t)
    t = re.sub(r'^V(?=\d)', "W", t)
    t = t.rstrip('/')  # strip trailing slash OCR artifact (e.g. "HSS6X6X3/8/")
    return t


_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}


def _pytesseract_to_dict(data: dict, page_w: float, page_h: float,
                         pix_w: int, pix_h: int) -> dict:
    """Convert pytesseract image_to_data output to a fitz-compatible text dict."""
    sx = page_w / max(pix_w, 1)
    sy = page_h / max(pix_h, 1)
    blocks = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0
        if conf < 30:        # skip very low-confidence words
            continue
        lx = data["left"][i]  * sx
        ty = data["top"][i]   * sy
        rx = (data["left"][i] + data["width"][i])  * sx
        by = (data["top"][i]  + data["height"][i]) * sy
        h  = max(by - ty, 1)
        blocks.append({
            "type": 0,
            "bbox": (lx, ty, rx, by),
            "lines": [{
                "bbox": (lx, ty, rx, by),
                "spans": [{"text": text, "bbox": (lx, ty, rx, by),
                            "size": h * 0.75}],
            }],
        })
    return {"blocks": blocks}


_easyocr_reader = None   # lazy-loaded singleton

def _get_easyocr_reader():
    """Return a cached EasyOCR reader (loaded once per process)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            print("[OCR] Loading EasyOCR model (first-run download may take a moment)...")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[OCR] EasyOCR ready")
        except Exception as e:
            print(f"[OCR] EasyOCR load failed: {e}")
    return _easyocr_reader


def _easyocr_to_dict(results: list, page_w: float, page_h: float,
                     pix_w: int, pix_h: int) -> dict:
    """
    Convert EasyOCR result list to fitz-compatible text dict.

    Each item in results may be a 3-tuple (bbox_pts, text, conf) or a
    4-tuple (bbox_pts, text, conf, rot_pass) where rot_pass is:
      0 = original orientation (horizontal text)
      1 = 90° CCW rotation (reads CW-rotated vertical text in original)
      2 = 90° CW rotation  (reads CCW-rotated vertical text in original)

    rot_pass is stored in each span so that extract_profiles can use it
    to infer beam direction for raster images (pass 1/2 → vertical member).

    All coordinates are in PIXEL space of the rendered image; we scale to
    fitz page-point space.
    """
    sx = page_w / max(pix_w, 1)
    sy = page_h / max(pix_h, 1)
    blocks = []
    for item in results:
        if len(item) == 4:
            bbox_pts, text, conf, rot_pass = item
        else:
            bbox_pts, text, conf = item
            rot_pass = 0
        text = text.strip()
        if not text or conf < 0.20:          # lowered from 0.25 for better recall
            continue
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        lx = min(xs) * sx;  rx = max(xs) * sx
        ty = min(ys) * sy;  by = max(ys) * sy
        h  = max(by - ty, 1)
        blocks.append({
            "type": 0,
            "bbox": (lx, ty, rx, by),
            "lines": [{
                "bbox": (lx, ty, rx, by),
                "spans": [{"text": text, "bbox": (lx, ty, rx, by),
                            "size": h * 0.75,
                            "rot_pass": rot_pass}],
            }],
        })
    return {"blocks": blocks}


def _get_text_dict(page, page_w: float = None, page_h: float = None
                   ) -> tuple[dict, bool]:
    """
    Return (text_dict, is_raster) for a fitz page.

    Attempt order
    ─────────────
    1. Embedded vector text (zero cost, always correct for proper PDFs)
    2. EasyOCR (pure-Python, no Tesseract required, good on engineering drawings)
    3. PyMuPDF built-in OCR  — requires Tesseract + tessdata
    4. pytesseract           — requires pytesseract + Tesseract binary

    Returns is_raster=False when embedded text was sufficient, True otherwise.
    """
    # ── 1. Embedded text ─────────────────────────────────────────────────────
    td = page.get_text("dict")
    n_chars = sum(len(sp["text"].strip())
                  for bl in td.get("blocks", [])
                  for ln in bl.get("lines", [])
                  for sp in ln.get("spans", []))
    if n_chars >= 20:
        return td, False

    print(f"[OCR] Embedded text chars={n_chars} — activating OCR")

    pw = page_w or page.rect.width
    ph = page_h or page.rect.height

    # ── 2. EasyOCR — 3-pass (0°, 90° CCW, 90° CW) + preprocessing ──────────────
    #
    # Structural drawings place beam labels PARALLEL to the member axis so
    # vertical member labels are rotated 90° in the image.  A single forward
    # pass only reads horizontal text; we rotate the image and transform
    # coordinates back to catch vertical text as well.
    #
    # Each result is stored as a 4-tuple (bbox, text, conf, rot_pass) where
    # rot_pass encodes which orientation found the text:
    #   0 = original  →  horizontal member label  (dir_hint = H)
    #   1 = 90° CCW   →  CW-rotated label in original  (dir_hint = V)
    #   2 = 90° CW    →  CCW-rotated label in original (dir_hint = V)
    #
    # Preprocessing: convert to greyscale, boost contrast, sharpen.
    # This significantly improves OCR accuracy on dense engineering drawings
    # with thin lines, small text, and low ink-to-paper contrast.
    reader = _get_easyocr_reader()
    if reader is not None:
        try:
            pix     = page.get_pixmap(dpi=300)        # was 200 — higher res for small text
            pil_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # CLAHE preprocessing: adaptive local contrast enhancement.
            # Better than global contrast because it improves dark/faded regions
            # without over-saturating bright ones, keeping thin strokes like
            # "/" in HSS profiles (e.g. HSS6X6X3/8) readable.
            try:
                import cv2 as _cv2
                _gray = np.array(pil_img.convert("L"))
                # Step 1: CLAHE for local contrast (handles uneven ink/scan)
                _clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                _enhanced = _clahe.apply(_gray)
                # Step 2: Unsharp mask to sharpen blurry scanned text.
                # Scanned drawings often have low-frequency blur from the scanner
                # optics; sharpening recovers thin strokes like "/" in "HSS6X6X3/8".
                _blur    = _cv2.GaussianBlur(_enhanced, (0, 0), sigmaX=1.5)
                _sharp   = _cv2.addWeighted(_enhanced, 1.5, _blur, -0.5, 0)
                pil_img = PILImage.fromarray(
                    _cv2.cvtColor(_sharp, _cv2.COLOR_GRAY2RGB))
            except Exception:
                # Fallback: mild global contrast if cv2 unavailable
                pil_img = PILImage.fromarray(
                    np.stack([np.array(ImageEnhance.Contrast(
                        pil_img.convert("L")).enhance(1.4))] * 3, axis=-1))

            img_arr = np.array(pil_img)
            H_pix, W_pix = img_arr.shape[:2]

            # Characters found in structural steel section labels + grid bubbles
            _ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./-×X"

            # EasyOCR parameters tuned for engineering drawings.
            # We intentionally keep link_threshold at the default (0.4) to
            # prevent characters from different nearby labels being merged
            # into one span (e.g. "W12X26" + adjacent "1" → "W12X261").
            # text_threshold lowered slightly to catch low-ink labels.
            _OCR_KW = dict(
                detail=1, allowlist=_ALLOW, paragraph=False,
                text_threshold=0.65,   # default 0.7 — slightly more detections
                low_text=0.35,         # default 0.4 — catch edge characters
                link_threshold=0.4,    # default 0.4 — keep to avoid cross-label merging
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )

            # Pass 0 — 0°: horizontal text (most beam labels, grid numbers)
            res = [(bbox, text, conf, 0)
                   for (bbox, text, conf) in reader.readtext(img_arr, **_OCR_KW)]

            # Pass 1 — 90° CCW: reads labels rotated 90° CW in the original.
            #   rot90 CCW transform: orig(x,y) → rot(x_r=y, y_r=W-1-x)
            #   Inverse: rot(x_r,y_r) → orig(x=W-1-y_r, y=x_r)
            img_rot = np.rot90(img_arr, k=1)
            for (bbox, text, conf) in reader.readtext(img_rot, **_OCR_KW):
                orig_bbox = [[W_pix - 1 - float(py), float(px)]
                             for (px, py) in bbox]
                res.append((orig_bbox, text, conf, 1))

            # Pass 2 — 90° CW: reads labels rotated 90° CCW in the original.
            #   rot90 CW (=rot270 CCW): orig(x,y) → rot(x_r=H-1-y, y_r=x)
            #   Inverse: rot(x_r,y_r) → orig(x=y_r, y=H-1-x_r)
            img_rot3 = np.rot90(img_arr, k=3)
            for (bbox, text, conf) in reader.readtext(img_rot3, **_OCR_KW):
                orig_bbox = [[float(py), H_pix - 1 - float(px)]
                             for (px, py) in bbox]
                res.append((orig_bbox, text, conf, 2))

            td2 = _easyocr_to_dict(res, pw, ph, W_pix, H_pix)
            n2  = sum(len(sp["text"].strip())
                      for bl in td2.get("blocks", [])
                      for ln in bl.get("lines", [])
                      for sp in ln.get("spans", []))
            if n2 >= 10:
                print(f"[OCR] EasyOCR 3-pass: {n2} chars / {len(res)} regions  "
                      f"(pass0={sum(1 for r in res if r[3]==0)}  "
                      f"pass1={sum(1 for r in res if r[3]==1)}  "
                      f"pass2={sum(1 for r in res if r[3]==2)})")
                return td2, True
        except Exception as e:
            print(f"[OCR] EasyOCR inference failed: {e}")

    # ── 3. PyMuPDF built-in OCR (Tesseract) ──────────────────────────────────
    try:
        tp  = page.get_textpage_ocr(flags=0, language="eng", dpi=300, full=True)
        td3 = page.get_text("dict", textpage=tp)
        n3  = sum(len(sp["text"].strip())
                  for bl in td3.get("blocks", [])
                  for ln in bl.get("lines", [])
                  for sp in ln.get("spans", []))
        if n3 >= 10:
            print(f"[OCR] PyMuPDF/Tesseract: {n3} chars extracted")
            return td3, True
    except Exception as e:
        print(f"[OCR] PyMuPDF OCR: {e}")

    # ── 4. pytesseract ────────────────────────────────────────────────────────
    try:
        import pytesseract as _pyt
        pix     = page.get_pixmap(dpi=200)
        pil_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        data    = _pyt.image_to_data(pil_img, output_type=_pyt.Output.DICT,
                                     lang="eng", config="--psm 11")
        td4  = _pytesseract_to_dict(data, pw, ph, pix.width, pix.height)
        n4   = sum(len(sp["text"].strip())
                   for bl in td4.get("blocks", [])
                   for ln in bl.get("lines", [])
                   for sp in ln.get("spans", []))
        if n4 >= 10:
            print(f"[OCR] pytesseract: {n4} chars extracted")
            return td4, True
    except Exception as e:
        print(f"[OCR] pytesseract: {e}")

    print("[OCR] All OCR methods failed for this raster image.")
    return {"blocks": []}, True


# ── Plan boundary (from grid bubble positions) ────────────────────────────────
def find_plan_boundary(page, page_w, page_h, text_dict=None):
    """
    Derive the structural plan extent from grid bubble positions.

    Grid bubbles (letter labels A/B/C… AND number labels 1/2/3…) always appear
    AROUND the perimeter of the structural plan.  Their collective bounding box
    plus a small buffer gives a reliable plan boundary regardless of drawing
    orientation.

    This handles BOTH common layouts transparently:
      • Letters at top/bottom, numbers at left/right   (e.g. "Structural snaps")
      • Letters at left/right, numbers at top/bottom   (e.g. Calsteel drawings)
      • Any mixed layout

    Why NOT separate letter_xs / number_ys (old approach):
      In a "letters-on-left" drawing all letter X values cluster near the left
      edge, so letter_xs gives a ~0-width X range — the plan boundary collapses
      to a sliver and nothing is extracted.

    Gap detection (Y axis):
      If the sorted Y positions of all grid labels have a gap > 12% of page
      height we drop everything below that gap.  Such a gap signals the end of
      the structural plan and the start of the notes / title block area.
    """
    all_xs: list[float] = []
    all_ys: list[float] = []

    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t  = span["text"].strip()
                fs = span.get("size", 10)
                if not t or len(t) > 5 or fs < 6:
                    continue
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2

                if _GRID_LETTER.match(t):
                    all_xs.append(cx)
                    all_ys.append(cy)
                elif _GRID_NUMBER.match(t):
                    try:
                        n = float(t)
                        # Accept grid numbers 0.5–500.  Upper bound was 25 which
                        # silently dropped columns numbered 26+ (e.g. a drawing
                        # spanning grids 21–34 lost its entire right half).
                        # 500 covers any practical building grid while still
                        # excluding large dimension/load annotation numbers.
                        if 0.5 <= n <= 500:
                            all_xs.append(cx)
                            all_ys.append(cy)
                    except ValueError:
                        pass

    # ── Gap detection on Y: drop labels below the first large vertical gap ─────
    # Large gap  ≡  notes / title block is separated from the structural plan.
    # Threshold raised to 20% so that widely-spaced bottom grid rows (e.g. A.7→A)
    # are NOT incorrectly trimmed — those rows still contain real structural members.
    if len(all_ys) >= 3:
        sorted_ys = sorted(all_ys)
        best_gap   = 0.0
        gap_cutoff = sorted_ys[-1]           # default: keep everything
        for i in range(len(sorted_ys) - 1):
            g = sorted_ys[i + 1] - sorted_ys[i]
            if g > best_gap:
                best_gap   = g
                gap_cutoff = sorted_ys[i]    # last Y before the gap
        if best_gap > page_h * 0.28:         # was 0.12 — raised; only fires for very obvious title-block separation
            old_max = max(all_ys)
            all_xs = [all_xs[i] for i, y in enumerate(all_ys) if y <= gap_cutoff]
            all_ys = [y           for y in all_ys               if y <= gap_cutoff]
            if all_ys:
                print(f"[BOUNDARY] gap={best_gap:.0f}pt — trimmed Y max "
                      f"from {old_max:.0f} to {max(all_ys):.0f}")

    if len(all_xs) >= 2 and len(all_ys) >= 2:
        # 300 pt buffer: raised from 150 so beams and labels at the outermost
        # column strip (which sits well beyond the last grid bubble) are included.
        # This is the universal fix for drawings where plan_bounds was clipping
        # the right / bottom edge of the framing area.
        buf = 300
        b = (
            max(0,      min(all_xs) - buf),
            max(0,      min(all_ys) - buf),
            min(page_w, max(all_xs) + buf),
            min(page_h, max(all_ys) + buf),
        )
        print(f"[BOUNDARY] grid-derived x=[{b[0]:.0f},{b[2]:.0f}] "
              f"y=[{b[1]:.0f},{b[3]:.0f}]")
        return b

    print("[BOUNDARY] fallback margins")
    return (page_w * 0.03, page_h * 0.02, page_w * 0.97, page_h * 0.98)


# ── Column symbol detection (I/H cross-section marks in vector drawings) ──────
def detect_column_symbols(page):
    """
    Detect the small I-section / W-section plan-view symbols drawn in the PDF.

    In structural steel framing plans, each column position is marked with a
    small I or H shaped symbol (two short horizontal lines joined by a vertical
    web line) representing the W-beam cross-section viewed from above.

    Strategy:
      1. Collect all drawing paths whose bounding box is small (5–50 PDF pts).
      2. For each small path, inspect its line segments for both horizontal AND
         vertical components — the defining characteristic of an I/H shape.
      3. Also accept rectangular paths (some CAD exports draw the column symbol
         as a filled or outlined small rectangle with flanges).
      4. Cluster nearby hits and deduplicate.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    raw = []   # (cx, cy) candidates

    def _ang_diff(a, b):
        """Smallest angle between two undirected lines (0–90°)."""
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _has_IH_pattern(angles, tol=20.0):
        """Rotation-invariant I/H test: ≥2 mutually-parallel segments (the two
        flanges) PLUS ≥1 segment roughly perpendicular to them (the web), at ANY
        orientation.  This recognises the column symbol whether it is drawn
        upright (flanges horizontal) OR rotated to any angle (skewed grids,
        canopy framing, angled wings)."""
        for a in angles:
            par  = sum(1 for b in angles if _ang_diff(a, b) <= tol)
            perp = sum(1 for b in angles if abs(_ang_diff(a, b) - 90.0) <= tol)
            if par >= 2 and perp >= 1:
                return True
        return False

    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        w, h = rect.width, rect.height

        # Column symbol bounding box: small in both axes
        # Too large = beam line or wall; too small = arrowhead or dimension tick
        if not (4 < w < 50 and 4 < h < 50):
            continue

        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2

        # ── Check drawing fill colour up-front ───────────────────────────────
        # Column plan marks are solid BLACK fills.
        # Dimension ticks / annotation boxes are un-filled (fill=None) or light.
        # We read the fill once here and use it inside the item loop.
        drawing_fill       = d.get("fill")
        drawing_brightness = 1.0   # assume light until proven dark
        if drawing_fill is not None and len(drawing_fill) >= 3:
            drawing_brightness = (drawing_fill[0] +
                                  drawing_fill[1] +
                                  drawing_fill[2]) / 3

        items     = d.get("items", [])
        has_curve = False
        n_lines   = 0
        h_count   = 0   # number of horizontal segments (flanges)
        v_count   = 0   # number of vertical segments (web)
        seg_angles = []  # angle (mod 180°) of every line segment — for the
                         # rotation-invariant I/H test below

        for item in items:
            kind = item[0]
            if kind == "c":
                has_curve = True   # arc/circle → callout bubble, not a column
                break
            elif kind == "l":
                n_lines += 1
                try:
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    if math.hypot(dx, dy) < 2:
                        continue
                    seg_angles.append(math.degrees(math.atan2(p2.y - p1.y,
                                                              p2.x - p1.x)) % 180.0)
                    if dx > dy * 1.5:
                        h_count += 1
                    elif dy > dx * 1.5:
                        v_count += 1
                except Exception:
                    continue
            elif kind == "re":
                # PDF rectangle primitive.
                # ONLY accept if the drawing has a VERY DARK (near-black) fill —
                # this is the signature of a structural column plan mark.
                # Un-filled outlines and light-coloured annotation boxes are ignored.
                if drawing_brightness < 0.25:
                    try:
                        rr = item[1]
                        rw = abs(rr.x1 - rr.x0)
                        rh = abs(rr.y1 - rr.y0)
                        ra = rw / rh if rh > 0 else 0
                        # Must be reasonably square and large enough to be a mark
                        if rw >= 4 and rh >= 4 and 0.25 < ra < 4.0:
                            h_count += 2
                            v_count += 1
                            n_lines += 4
                            seg_angles += [0.0, 0.0, 90.0]  # box = 2 flanges + web
                    except Exception:
                        pass

        aspect = w / h if h > 0 else 0

        # ── Accept I/H shape (rotation-invariant) ────────────────────────────
        # ≥2 parallel flange segments + ≥1 perpendicular web segment at ANY
        # orientation.  This detects the column symbol whether it is upright OR
        # rotated to any angle (skewed grids, canopy/angled framing) — the case
        # where the old strict horizontal/vertical test missed columns, leaving
        # beams with no centre to snap to.
        if not has_curve and _has_IH_pattern(seg_angles):
            raw.append((cx, cy))
            continue

        # ── Accept small FILLED rectangle (some CAD exports use a solid box) ──
        # Previously accepted ANY 4-line outlined rectangle, which caught
        # connection plates, stiffener boxes, joist seats and dimension ticks —
        # all legitimate drawing elements that are NOT column symbols.
        # Now require near-black fill (brightness < 0.25): only solid black
        # plan marks qualify.  Un-filled or lightly-shaded boxes are skipped.
        if (not has_curve and n_lines == 4 and 0.5 < aspect < 2.0
                and w < 28 and h < 28 and drawing_brightness < 0.25):
            raw.append((cx, cy))
            continue

        # ── Accept small CIRCLED I/H symbol ───────────────────────────────────
        # Many CAD drawings ring the column I/H mark with a small circle.
        # Grid bubbles are large (>45pt); section-cut bubbles rarely have both
        # H and V lines inside them. Limit to bbox 10–45pt to stay specific.
        if has_curve and _has_IH_pattern(seg_angles) and 10 < w < 45 and 10 < h < 45:
            raw.append((cx, cy))

    # Deduplicate using EXPANDING cluster (DBSCAN-style, eps=15pt).
    #
    # WHY NOT a fixed radius:
    #   • Some CAD exports draw each column symbol as 6-8 separate filled-rectangle
    #     sub-paths (one per flange/web piece). Sub-paths of the SAME symbol are
    #     typically 8-15pt apart from each other.
    #   • A simple fixed-radius check from the SEED only merges points within that
    #     radius of the first point. If the cluster spans 70pt but each step is
    #     only 10pt, a 12pt seed-radius misses outer sub-paths.
    #   • A 60pt fixed radius merges real adjacent columns when the drawing scale
    #     is small (columns can be 40-50pt apart on some sheets).
    #
    # EXPANDING CLUSTER solution:
    #   eps=15pt — small enough to never bridge two real columns (always >30pt apart
    #   at any typical drawing scale), large enough to chain sub-paths of the same
    #   symbol together step-by-step regardless of total symbol span.
    EPS = 15
    symbols = []
    used = [False] * len(raw)
    for i in range(len(raw)):
        if used[i]:
            continue
        # Seed the cluster with point i
        cluster_idx = [i]
        used[i] = True
        # Expand: keep sweeping until no new neighbours are added
        queue = [i]
        while queue:
            cur_idx = queue.pop()
            cx_cur, cy_cur = raw[cur_idx]
            for j in range(len(raw)):
                if not used[j]:
                    if math.hypot(cx_cur - raw[j][0], cy_cur - raw[j][1]) < EPS:
                        used[j] = True
                        cluster_idx.append(j)
                        queue.append(j)
        xs = [raw[k][0] for k in cluster_idx]
        ys = [raw[k][1] for k in cluster_idx]
        symbols.append({
            "cx": sum(xs) / len(xs),
            "cy": sum(ys) / len(ys),
        })

    print(f"[SYMBOLS] Column symbols detected: {len(symbols)}")
    return symbols


def detect_raster_grid_lines(img_path: str, plan_bounds: tuple) -> tuple:
    """
    Detect structural column grid lines from a raster image file.

    Uses morphological opening with long kernels:
      • Vertical kernel (height ≥ 40% of plan)  → isolates tall vertical lines
        → their X positions become v_grid
      • Horizontal kernel (width ≥ 40% of plan) → isolates wide horizontal lines
        → their Y positions become h_grid

    Structural column lines span the full plan height/width; beam centerlines,
    annotation lines, and other short marks are suppressed.
    """
    try:
        import cv2 as _cv2
        from PIL import Image as _PIL
    except ImportError:
        return [], []

    try:
        pil = _PIL.open(img_path).convert("RGB")
        img_arr = np.array(pil)
    except Exception:
        return [], []

    bx0, by0, bx1, by1 = [int(round(x)) for x in plan_bounds]
    bx0 = max(bx0, 0); by0 = max(by0, 0)
    bx1 = min(bx1, img_arr.shape[1]); by1 = min(by1, img_arr.shape[0])
    H = by1 - by0; W = bx1 - bx0
    if H < 50 or W < 50:
        return [], []

    region = img_arr[by0:by1, bx0:bx1]
    gray = _cv2.cvtColor(region, _cv2.COLOR_RGB2GRAY)
    # Invert: dark structural lines → white in binary
    _, binary = _cv2.threshold(gray, 200, 255, _cv2.THRESH_BINARY_INV)

    def _cluster_avg(vals: list, tol: float = 12.0) -> list:
        """Merge nearby positions by averaging each cluster."""
        if not vals:
            return []
        result = []
        group = [vals[0]]
        for v in sorted(vals)[1:]:
            if v - group[-1] <= tol:
                group.append(v)
            else:
                result.append(sum(group) / len(group))
                group = [v]
        result.append(sum(group) / len(group))
        return result

    # ── Vertical grid lines (column lines running top→bottom) ──────────────
    v_klen = max(30, int(H * 0.40))
    v_kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (1, v_klen))
    v_img = _cv2.morphologyEx(binary, _cv2.MORPH_OPEN, v_kernel)
    v_sums = np.sum(v_img, axis=0).astype(float) / 255
    v_threshold = H * 0.25
    raw_vx = [float(x + bx0) for x in range(W) if v_sums[x] >= v_threshold]
    v_grid = _cluster_avg(raw_vx, tol=15)

    # ── Horizontal grid lines (row lines running left→right) ───────────────
    h_klen = max(30, int(W * 0.40))
    h_kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (h_klen, 1))
    h_img = _cv2.morphologyEx(binary, _cv2.MORPH_OPEN, h_kernel)
    h_sums = np.sum(h_img, axis=1).astype(float) / 255
    h_threshold = W * 0.25
    raw_hy = [float(y + by0) for y in range(H) if h_sums[y] >= h_threshold]
    h_grid = _cluster_avg(raw_hy, tol=15)

    print(f"[RASTER_GRID] V({len(v_grid)}): {[round(x) for x in v_grid]}")
    print(f"[RASTER_GRID] H({len(h_grid)}): {[round(y) for y in h_grid]}")
    return v_grid, h_grid


def detect_beam_lines_raster(img_path: str, profiles: list,
                             plan_bounds: tuple,
                             span_v_grid: list = None,
                             span_h_grid: list = None) -> dict:
    """
    Raster equivalent of detect_beam_lines for vector PDFs.

    Uses actual drawn Hough line segments as span endpoints — each beam gets
    exactly the length of its drawn line in the image, just like vector PDF
    mode reads CAD geometry.

    Max-span cap is derived dynamically from the column-bay widths so that
    full-plan column-grid lines and sheet borders are always excluded.

    Returns { profile_idx: {"x1","y1","x2","y2","dir","length_pt"} }
    """
    try:
        import cv2 as _cv2
        from PIL import Image as _PIL
    except ImportError:
        return {}

    try:
        pil = _PIL.open(img_path).convert("L")
        img_gray = np.array(pil)
    except Exception:
        return {}

    bx0, by0, bx1, by1 = [int(round(x)) for x in plan_bounds]
    bx0 = max(bx0, 0); by0 = max(by0, 0)
    bx1 = min(bx1, img_gray.shape[1]); by1 = min(by1, img_gray.shape[0])
    plan_w = bx1 - bx0; plan_h = by1 - by0
    if plan_w < 50 or plan_h < 50:
        return {}

    # Derive dynamic max-span from bay widths so column/border lines are excluded.
    # Cap = 1.4× the widest detected structural bay (allows slight overrun).
    vg = span_v_grid or []
    hg = span_h_grid or []

    if len(vg) >= 2:
        max_bay_w = max(b - a for a, b in zip(sorted(vg), sorted(vg)[1:]))
        MAX_H = max_bay_w * 1.4
    else:
        MAX_H = plan_w * 0.45

    if len(hg) >= 2:
        max_bay_h = max(b - a for a, b in zip(sorted(hg), sorted(hg)[1:]))
        MAX_V = max_bay_h * 1.4
    else:
        MAX_V = plan_h * 0.45

    region = img_gray[by0:by1, bx0:bx1]
    edges = _cv2.Canny(region, threshold1=50, threshold2=150, apertureSize=3)
    raw = _cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=25,
        minLineLength=45,
        maxLineGap=6,          # small gap: don't bridge across column locations
    )
    if raw is None:
        return {}

    DIR_R   = 2.5
    LABEL_R = 60    # px — was 30; EasyOCR offset + Hough quantisation stacks to ~35 px;
                    # 60 px covers both error sources without jumping a full bay
    MIN_LEN = 45

    h_lines: list[tuple] = []   # (x1, y, x2, y, length)
    v_lines: list[tuple] = []   # (x, y1, x, y2, length)
    d_lines: list[tuple] = []   # (x1, y1, x2, y2, length)

    for seg in raw:
        x1r, y1r, x2r, y2r = seg[0]
        x1 = float(x1r + bx0); y1 = float(y1r + by0)
        x2 = float(x2r + bx0); y2 = float(y2r + by0)
        dx = abs(x2 - x1); dy = abs(y2 - y1)
        ln = math.hypot(dx, dy)
        if ln < MIN_LEN:
            continue
        if dx > dy * DIR_R and ln <= MAX_H:
            my = (y1 + y2) / 2
            h_lines.append((min(x1, x2), my, max(x1, x2), my, ln))
        elif dy > dx * DIR_R and ln <= MAX_V:
            mx = (x1 + x2) / 2
            v_lines.append((mx, min(y1, y2), mx, max(y1, y2), ln))
        else:
            if ln <= max(MAX_H, MAX_V):
                d_lines.append((x1, y1, x2, y2, ln))

    print(f"[RASTER_BEAM] MAX_H={MAX_H:.0f}px MAX_V={MAX_V:.0f}px  "
          f"H lines: {len(h_lines)}  V lines: {len(v_lines)}  D lines: {len(d_lines)}")

    result: dict = {}
    for p_idx, p in enumerate(profiles):
        pcx, pcy = p["cx"], p["cy"]
        best = None; best_d = float("inf"); best_dir = "H"

        # Prefer longer lines when equidistant (longer = more likely real beam)
        for (lx1, ly, lx2, _, ln) in h_lines:
            dy_l = abs(pcy - ly)
            if dy_l > LABEL_R:
                continue
            if pcx < lx1 - LABEL_R or pcx > lx2 + LABEL_R:
                continue
            score = dy_l - ln * 0.01   # slight length bonus
            if score < best_d:
                best_d = score
                best = (lx1, ly, lx2, ly, ln)
                best_dir = "H"

        for (lx, ly1, _, ly2, ln) in v_lines:
            dx_l = abs(pcx - lx)
            if dx_l > LABEL_R:
                continue
            if pcy < ly1 - LABEL_R or pcy > ly2 + LABEL_R:
                continue
            score = dx_l - ln * 0.01
            if score < best_d:
                best_d = score
                best = (lx, ly1, lx, ly2, ln)
                best_dir = "V"

        for (lx1, ly1, lx2, ly2, ln) in d_lines:
            # point-to-line distance
            den = math.hypot(lx2 - lx1, ly2 - ly1)
            num = abs((lx2 - lx1) * (ly1 - pcy) - (lx1 - pcx) * (ly2 - ly1))
            dist = num / den if den > 0 else float('inf')
            if dist > LABEL_R:
                continue
            # check if projection is within segment
            dot = ((pcx - lx1)*(lx2 - lx1) + (pcy - ly1)*(ly2 - ly1)) / (den*den) if den > 0 else -1
            if dot < -0.1 or dot > 1.1:
                continue
            score = dist - ln * 0.01
            if score < best_d:
                best_d = score
                best = (lx1, ly1, lx2, ly2, ln)
                best_dir = "D"

        if best:
            result[p_idx] = {
                "x1": best[0], "y1": best[1],
                "x2": best[2], "y2": best[3],
                "dir": best_dir,
                "length_pt": best[4],
            }

    print(f"[RASTER_BEAM] {len(result)}/{len(profiles)} profiles matched")
    return result


# ── Grid line extraction ──────────────────────────────────────────────────────
def extract_grid_lines(page, page_w, page_h, plan_bounds, text_dict=None):
    """
    Extract the X positions of vertical grid lines and the Y positions of
    horizontal grid lines from grid bubble label text.

    Orientation-agnostic strategy
    ─────────────────────────────
    Rather than assuming "letters are at top/bottom" or "numbers are at
    left/right" (which breaks on Calsteel-style drawings that put letters on
    the left and numbers at the top), we measure the SPREAD of each label
    family within the plan boundary:

      • If the label family spans more in X than in Y → the bubbles form a
        horizontal row at the top or bottom of the plan → they label VERTICAL
        grid lines → contribute their X positions to v_raw.

      • If the family spans more in Y than in X → the bubbles form a vertical
        column at the left or right → they label HORIZONTAL grid lines →
        contribute their Y positions to h_raw.

    This works correctly for both common layouts and any mixed-orientation plan.
    """
    bx0, by0, bx1, by1 = plan_bounds

    # Collect positions of each label family inside the plan boundary.
    #
    # We only trust labels that sit near the PERIMETER of the plan (outer 25%).
    # Grid bubbles are always at the top/bottom/left/right edge — never in the
    # interior.  Interior annotation numbers (joist loads "18K" OCR'd as "18",
    # span callouts "22", etc.) would otherwise pollute v_grid / h_grid and
    # make every beam span collapse to near-zero length.
    plan_w = max(bx1 - bx0, 1.0)
    plan_h = max(by1 - by0, 1.0)

    # ── Dynamic EDGE zone ────────────────────────────────────────────────────
    # Standard drawings: plan is 50-75% of page → EDGE=0.25 captures all
    # grid bubbles at the perimeter safely.
    #
    # Full-page drawings (Calsteel style, large hospitals): plan occupies
    # 90-100% of the page.  EDGE=0.25 only captures the outermost rows —
    # e.g. rows A-E and N-R of an A-R plan (18 rows), missing rows F-M.
    # Missing rows → wrong grid → wrong span fallback → chips in wrong place.
    #
    # Formula: EDGE scales from 0.25 (plan ≤ 70% of page) up to 0.45
    # (plan = full page).  This widens the capture zone for large plans
    # WITHOUT changing behaviour for standard-sized drawings.
    plan_cov = max((bx1 - bx0) / max(page_w, 1.0),
                   (by1 - by0) / max(page_h, 1.0))
    # Kicks in only when plan_cov > 0.70 to avoid widening on normal drawings
    EDGE = min(0.45, 0.25 + max(0.0, plan_cov - 0.70) * 0.67)

    letter_pts: list[tuple[float, float]] = []
    number_pts: list[tuple[float, float]] = []

    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t  = span["text"].strip()
                fs = span.get("size", 10)
                # Require minimum legible font size for grid bubbles.
                # Interior dimension annotations are typically drawn at < 7 pt;
                # grid bubble text is always ≥ 7 pt (must be readable on plot).
                if not t or len(t) > 5 or fs < 7:
                    continue
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2
                # Must be inside (or very close to) the plan boundary
                if not (bx0 - 60 <= cx <= bx1 + 60 and
                        by0 - 60 <= cy <= by1 + 60):
                    continue
                # Must be near the perimeter — grid bubbles are always at the edge
                near_edge = (
                    cx < bx0 + plan_w * EDGE or cx > bx1 - plan_w * EDGE or
                    cy < by0 + plan_h * EDGE or cy > by1 - plan_h * EDGE
                )
                if not near_edge:
                    continue

                if _GRID_LETTER.match(t):
                    letter_pts.append((cx, cy))
                elif _GRID_NUMBER.match(t):
                    try:
                        if not (0.5 <= float(t) <= 200):
                            continue
                    except ValueError:
                        continue
                    number_pts.append((cx, cy))

    def _classify_family(pts):
        """Return (v_positions, h_positions) for a set of bubble label points."""
        if not pts:
            return [], []
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        def _uniq(vals, tol=20):
            seen = []
            for v in sorted(vals):
                if not seen or abs(v - seen[-1]) > tol:
                    seen.append(v)
            return len(seen)
        if _uniq(xs) >= _uniq(ys):
            return xs, []
        else:
            return [], ys

    def _dominant_sequence(vals, span):
        """
        Universal interior-annotation filter.

        With a wider EDGE zone (needed for large/full-page plans), some
        interior annotation numbers (bay-span callouts like '18', '24')
        can enter the candidate set alongside real grid bubble labels.

        Real grid bubbles form a single CONTINUOUS sequence across the full
        plan span — no large gaps between them.  Interior annotation clusters
        sit at isolated positions separated from the main grid by large empty
        stretches.

        Algorithm: find the longest group of positions where consecutive
        entries are ≤ max_gap apart.  Return that group if it accounts for
        ≥ 55% of total candidates (i.e. real grid dominates); otherwise
        return all values unchanged (sparse / irregular grid, no filtering).

        max_gap = max(30% of plan span, 120 pt) — generous enough to handle
        any structural bay size (typical largest bay ≤ 30 ft = 270–405 pt).
        """
        if len(vals) < 3:
            return vals
        sv      = sorted(vals)
        max_gap = max(span * 0.30, 120.0)

        best, cur = [sv[0]], [sv[0]]
        for v in sv[1:]:
            if v - cur[-1] <= max_gap:
                cur.append(v)
            else:
                if len(cur) > len(best):
                    best = cur[:]
                cur = [v]
        if len(cur) > len(best):
            best = cur

        # Only apply if the dominant sequence is clearly the majority.
        # For truly sparse grids (large irregular bays) the condition won't
        # fire and all values are returned untouched.
        return best if len(best) >= max(3, len(vals) * 0.55) else vals

    lv, lh = _classify_family(letter_pts)
    nv, nh = _classify_family(number_pts)

    v_raw = lv + nv
    h_raw = lh + nh

    # Apply dominant-sequence filter to drop isolated annotation clusters
    # that sneak in when EDGE is widened for large / full-page drawings.
    v_raw = _dominant_sequence(v_raw, plan_w)
    h_raw = _dominant_sequence(h_raw, plan_h)

    def dedup(vals, tol=18):
        out = []
        for v in sorted(vals):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out

    v_grid = dedup(v_raw)
    h_grid = dedup(h_raw)
    print(f"[GRID] V({len(v_grid)}): {[round(x) for x in v_grid]}")
    print(f"[GRID] H({len(h_grid)}): {[round(y) for y in h_grid]}")
    return v_grid, h_grid


# ── Member classification ─────────────────────────────────────────────────────
def classify_member(profile: str,
                    cx: float = 0, cy: float = 0,
                    column_symbols: list = None,
                    v_grid: list = None,
                    h_grid: list = None) -> str:
    """
    Classify a steel section label as column / beam / brace.

    Works for ANY drawing — ordered from most reliable to least:

      TIER 0  Profile type   PIPE, square HSS → always column
      TIER 1  Symbol nearby  I/H mark detected close to this label
      TIER 2  Grid cross     label sits near a named grid intersection
      TIER 3  Depth rule     W6/W8/W10 are almost always columns in practice
      TIER 4  Weight rule    heavy section for its depth → column

    Keeping TIER 3 (depth rule) separate ensures short W-sections are caught
    even on drawings where symbol detection or grid detection yields nothing.
    """
    p = profile.upper().strip()

    # ── Angle / brace sections ────────────────────────────────────────────────
    if re.match(r'ISA', p) or re.match(r'L\d', p):
        return "brace"

    # ── TIER 0 — Profile type: always a column regardless of drawing ──────────
    if re.match(r'PIPE', p):
        return "column"

    hss = re.match(r'HSS([\d.]+)[Xx]([\d.]+)', p)
    if hss:
        try:
            d1 = float(hss.group(1))
            d2 = float(hss.group(2))
            # Only perfectly square HSS (e.g. HSS6X6, HSS8X8) are columns;
            # all rectangular HSS spanning between grids are beams.
            return "column" if abs(d1 - d2) < 0.5 else "beam"
        except ValueError:
            return "beam"

    # Abbreviated W-section (depth only — no weight in label, e.g. "W12", "W16")
    w_abbr = re.match(r'(?<![A-Z0-9])W(\d+)$', p)
    if w_abbr:
        # No weight info → treat as beam by default; beam-line override in
        # build_members will further validate against matched vector lines.
        return "beam"

    w = re.match(r'W(\d+)[Xx](\d+)', p)
    if w:
        depth  = int(w.group(1))
        weight = int(w.group(2))

        # Weight-to-depth column thresholds — used by ALL tiers so that a light
        # section (W12X26) near a column symbol is never falsely promoted.
        # A profile only qualifies as a column if its weight meets or exceeds
        # the threshold for its depth.
        _col_thresholds = {
            6:  0,    # W6  — always column
            7:  0,
            8:  0,    # W8  — always column
            9:  0,
            10: 22,   # W10×22+  → column
            12: 40,   # W12×40+  → column  (W12×26/30 stay beam)
            14: 38,   # W14×38+  → column  (W14×22/26/30 stay beam)
            16: 57,   # W16×57+  (W16×26/31/36/40 stay beam)
            18: 71,   # W18×71+  (W18×35/40/46/50 stay beam)
            21: 83,
            24: 94,
            27: 102,
        }
        _is_col_weight = weight >= _col_thresholds.get(depth, 9999)

        # ── TIER 1: near a detected I/H column symbol ─────────────────────────
        # The symbol drawn on the plan is the strongest signal — use it first.
        # BUT only promote to column if the section weight also confirms it.
        # Short beams framing INTO a column have their label placed right next
        # to the column symbol — without the weight check, they get stolen.
        if column_symbols:
            for sym in column_symbols:
                if math.hypot(cx - sym["cx"], cy - sym["cy"]) < SYMBOL_ASSOC_RADIUS:
                    if _is_col_weight or depth <= 10:
                        return "column"
                    # Light section near symbol = beam framing into column
                    break

        # ── TIER 2: at a named grid intersection ──────────────────────────────
        # Columns sit exactly at grid line crossings; beams span between them.
        # Same weight guard: a W12X26 at a grid crossing is a beam, not a column.
        if v_grid and h_grid:
            near_v = any(abs(cx - gx) < GRID_TOL for gx in v_grid)
            near_h = any(abs(cy - gy) < GRID_TOL for gy in h_grid)
            if near_v and near_h and _is_col_weight:
                return "column"

        # ── TIER 3: depth rule — short W-sections are columns on every drawing ─
        # W6 and W8 are almost never used as beams in structural framing plans.
        # W10 sections are columns far more often than beams.
        # This rule fires even when no symbols or grid are detected.
        if depth <= 8:
            return "column"
        if depth == 10 and weight >= 22:
            return "column"

        # ── TIER 4: weight-per-depth heuristic ────────────────────────────────
        # For W12/W14 (used both ways) and larger sections:
        # a high weight-to-depth ratio signals a compact column section.
        if _is_col_weight:
            return "column"

        return "beam"

    return "beam"


# ── Schedule / legend table exclusion ────────────────────────────────────────
def detect_schedule_zones(page, plan_bounds, text_dict=None):
    """
    Detect zones that are purely schedule/legend tables (not part of the
    structural plan) so we don't extract phantom members from them.

    Approach: look for steel-section labels that are OUTSIDE the plan boundary
    entirely (i.e., in the title block, notes column, or border area).
    The plan boundary already correctly clips the extraction region, so we
    only need to exclude labels that are JUST inside the boundary but clearly
    belong to tabular data — identified by extremely tight x-spread (< 8 pt,
    meaning they share almost the exact same x) AND a large y-span (> 250 pt)
    AND a very high label density (≥ 25 labels).

    Previously this used a 12-hit / 30-pt / 150-pt threshold which falsely
    excluded entire column lines on complex framing plans where many beams
    frame into the same grid line, producing a dense vertical label cluster
    that looks superficially like a schedule table.
    """
    bx0, by0, bx1, by1 = plan_bounds

    hits = []
    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2
                # Scan whole page so schedule tables outside plan_bounds are caught
                for pat in STEEL_PATTERNS:
                    if re.search(pat, text, re.IGNORECASE):
                        hits.append((cx, cy))
                        break

    if len(hits) < 10:
        return []

    excluded = []
    used = [False] * len(hits)

    for i, (ax, ay) in enumerate(hits):
        if used[i]:
            continue
        cluster = [(ax, ay)]
        used[i] = True
        for j, (bx, by) in enumerate(hits):
            if not used[j] and abs(ax - bx) < 8:   # very tight: < 8pt x-spread
                cluster.append((bx, by))
                used[j] = True

        # Require 25+ labels in that tiny x-band — only a true printed schedule
        # has this many entries at virtually the same x position.
        # Structural framing plans never concentrate this many beam labels on a
        # single column line with < 8pt x variation.
        if len(cluster) < 25:
            continue

        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)

        if x_spread < 8 and y_spread > 250:
            zone = (min(xs) - 60, min(ys) - 40, max(xs) + 60, max(ys) + 40)
            excluded.append(zone)
            print(f"[SCHEDULE] Col-cluster x=[{min(xs):.0f},{max(xs):.0f}] "
                  f"y=[{min(ys):.0f},{max(ys):.0f}] n={len(cluster)}")



    return excluded


# ── Profile extraction (within plan boundary only) ────────────────────────────
def extract_profiles(page, page_w, page_h, plan_bounds, text_dict=None):
    """
    Extract steel section labels from within the plan boundary.

    CAD PDFs frequently split a label like 'W18X40' across two text spans
    ('W18' and 'X40').  We match at both span-level AND full-line-level to
    catch every case, then deduplicate by proximity.
    """
    bx0, by0, bx1, by1 = plan_bounds
    excluded_zones = detect_schedule_zones(page, plan_bounds, text_dict=text_dict)
    profiles, seen = [], []

    def _try_add(text, cx, cy, rot_pass=0, bbox_w=20.0, bbox_h=8.0, angle=0.0):
        if not text:
            return
        # Scan the ENTIRE page for beam labels — do NOT filter by plan_bounds here.
        # Previously _LABEL_EDGE=120pt caused the entire right (or top/bottom) half
        # of a drawing to be silently dropped whenever plan_bounds was slightly
        # mis-detected.  STEEL_PATTERNS regex is specific enough that title-block
        # text never matches; schedule tables are excluded by detect_schedule_zones.
        # The label→line matcher (LABEL_R=130pt) and the structural length filter
        # act as the real proximity guard — no boundary pre-filter needed.
        _LABEL_EDGE = max(page_w, page_h)   # whole-page scan
        if not (bx0 - _LABEL_EDGE <= cx <= bx1 + _LABEL_EDGE and
                by0 - _LABEL_EDGE <= cy <= by1 + _LABEL_EDGE):
            return
        if any(zx0 <= cx <= zx1 and zy0 <= cy <= zy1
               for zx0, zy0, zx1, zy1 in excluded_zones):
            return
        for pat in STEEL_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                # Dedup radius 10 pt (was 20): dense framing plans have labels
                # as close as 10-15 pt; 20 pt dropped every second label.
                if not any(math.hypot(cx - sx, cy - sy) < 10
                           for sx, sy in seen):
                    seen.append((cx, cy))
                    # Three signals to determine vertical orientation:
                    # 1. rot_pass in (1,2): OCR found rotated text
                    # 2. |angle| > 70°: span["dir"] is near-vertical (≈90°).
                    #    IMPORTANT: use 70°, NOT 45°.  Diagonal beams in
                    #    rotated-grid drawings have labels at 30–60°; a 45°
                    #    threshold misclassifies them as vertical, swaps their
                    #    bbox, and then breaks the direction guard in
                    #    detect_beam_lines.  Only genuine 90°-rotated Revit
                    #    labels should trip this signal.
                    # 3. bbox_h > bbox_w*1.5: tall bbox = Revit pre-rotation
                    _is_vert = (rot_pass in (1, 2)
                                or abs(angle) > 70.0
                                or bbox_h > bbox_w * 1.5)
                    # Swap bbox dims for Revit pre-rotation: rotated text is
                    # stored with pre-rotation dimensions (wide instead of tall).
                    if _is_vert and bbox_w > bbox_h:
                        eff_w, eff_h = bbox_h, bbox_w
                    else:
                        eff_w, eff_h = bbox_w, bbox_h
                    profiles.append({
                        "profile": normalize_profile(m.group(0)),
                        "cx": cx, "cy": cy,
                        "dir_hint": "V" if _is_vert else "H",
                        "text_angle": angle,
                        # bbox dimensions — used by detect_beam_lines to infer
                        # expected beam direction (wide text → H beam; tall → V)
                        "bbox_w": max(eff_w, 1.0),
                        "bbox_h": max(eff_h, 1.0),
                    })
                break

    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])

            # ── Span-level: precise position per span ─────────────────────
            for span in spans:
                bbox = span["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                _dx, _dy = span.get("dir", (1.0, 0.0))
                _ang = math.degrees(math.atan2(-_dy, _dx))
                _try_add(span["text"].strip(), cx, cy,
                         rot_pass=span.get("rot_pass", 0),
                         bbox_w=bbox[2] - bbox[0],
                         bbox_h=bbox[3] - bbox[1],
                         angle=_ang)

            # ── Line-level: join all spans → catches "W18" + "X40" splits ─
            line_text = "".join(s["text"] for s in spans).strip()
            if line.get("bbox") and line_text:
                lb = line["bbox"]
                lx = (lb[0] + lb[2]) / 2
                ly = (lb[1] + lb[3]) / 2
                first_pass = spans[0].get("rot_pass", 0) if spans else 0
                if spans:
                    _ldx, _ldy = spans[0].get("dir", (1.0, 0.0))
                    _lang = math.degrees(math.atan2(-_ldy, _ldx))
                else:
                    _lang = 0.0
                _try_add(line_text, lx, ly, rot_pass=first_pass,
                         bbox_w=lb[2] - lb[0], bbox_h=lb[3] - lb[1],
                         angle=_lang)

    print(f"[EXTRACT] {len(profiles)} profiles found in plan")
    return profiles


def precompute_span_grids(profiles: list, plan_bounds: tuple,
                          page_w: float, page_h: float) -> tuple:
    """
    Quick pre-pass: classify profiles without grid context to find columns,
    then build span grids from their positions.

    Used by the /analyse endpoint to generate reliable span grids BEFORE
    detect_beam_lines_raster and build_members are called, so both can use
    the same accurate column-position-based grids.

    Returns (span_v_grid, span_h_grid) as sorted lists of pixel positions.
    """
    _pb = plan_bounds or (0, 0, page_w, page_h)

    col_xs: list[float] = []
    col_ys: list[float] = []
    for p in profiles:
        mt = classify_member(p["profile"], p["cx"], p["cy"],
                             column_symbols=None, v_grid=None, h_grid=None)
        if mt == "column":
            col_xs.append(p["cx"])
            col_ys.append(p["cy"])

    def _dedup(vals: list, tol: float = 20.0) -> list:
        out: list = []
        for v in sorted(vals):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out

    svg = _dedup(col_xs) if col_xs else []
    shg = _dedup(col_ys) if col_ys else []

    # Always include plan-boundary edges so edge beams get a span line
    svg = _dedup(sorted(svg + [_pb[0], _pb[2]]))
    shg = _dedup(sorted(shg + [_pb[1], _pb[3]]))

    print(f"[SPAN_GRIDS] V({len(svg)}): {[round(x) for x in svg]}")
    print(f"[SPAN_GRIDS] H({len(shg)}): {[round(y) for y in shg]}")
    return svg, shg


def _snap_endpoint_to_column(x: float, y: float,
                              column_symbols: list,
                              snap_radius: float = 90.0) -> tuple[float, float]:
    """Return the position of the nearest column symbol within snap_radius."""
    if not column_symbols:
        return x, y
    best = min(column_symbols, key=lambda s: math.hypot(s["cx"] - x, s["cy"] - y))
    if math.hypot(best["cx"] - x, best["cy"] - y) <= snap_radius:
        return best["cx"], best["cy"]
    return x, y


def _column_pair_span(cx: float, cy: float, beam_dir: str,
                      column_symbols: list, pts_per_foot: float,
                      max_span_pt: float = 380.0) -> dict | None:
    """
    Find the two nearest column symbols that a beam label sits BETWEEN and
    return span endpoints (exact column-to-column positions).

    max_span_pt caps the result so multi-bay false matches are rejected.
    At 1/8"=1'-0" scale, 380 pt ≈ 42 ft — typical max single bay.
    """
    if not column_symbols:
        return None

    if beam_dir in ("H", "D"):
        for tol in (25, 50, 80):
            row    = [s for s in column_symbols if abs(s["cy"] - cy) <= tol]
            lefts  = [s for s in row if s["cx"] < cx - 5]
            rights = [s for s in row if s["cx"] > cx + 5]
            if lefts and rights:
                c1 = max(lefts,  key=lambda s: s["cx"])
                c2 = min(rights, key=lambda s: s["cx"])
                break
        else:
            return None
    else:  # V
        for tol in (25, 50, 80):
            col     = [s for s in column_symbols if abs(s["cx"] - cx) <= tol]
            tops    = [s for s in col if s["cy"] < cy - 5]
            bottoms = [s for s in col if s["cy"] > cy + 5]
            if tops and bottoms:
                c1 = max(tops,    key=lambda s: s["cy"])
                c2 = min(bottoms, key=lambda s: s["cy"])
                break
        else:
            return None

    length_pt = math.hypot(c2["cx"] - c1["cx"], c2["cy"] - c1["cy"])
    if length_pt > max_span_pt:
        return None  # multi-bay false match — let grid fallback handle it
    return {
        "x1": c1["cx"], "y1": c1["cy"],
        "x2": c2["cx"], "y2": c2["cy"],
        "length_ft": round(length_pt / pts_per_foot, 1) if pts_per_foot > 0 else 0.0,
    }


# ── Build members + summary ───────────────────────────────────────────────────
def _span_valid(bx1f, by1f, bx2f, by2f, lx_frac, ly_frac, beam_dir) -> bool:
    """
    Sanity-check: the label must lie on (or very near) the computed span line.

    Why this matters
    ----------------
    For vector PDFs detect_beam_lines() guarantees this by construction.
    For the grid-based fallback (compute_beam_span), wrong grid-line matches
    can produce span endpoints that are on the opposite side of the plan from
    the label — the line appears visually far from the chip on screen.

    Rules (all tolerances as fractions of the page dimension):
      H-beam  — span Y must be within PERP_TOL of label Y
                 label X must fall inside [bx1−EXT_TOL, bx2+EXT_TOL]
      V-beam  — span X must be within PERP_TOL of label X
                 label Y must fall inside [by1−EXT_TOL, by2+EXT_TOL]
    """
    # PERP_TOL: how far (as fraction of page) the label can sit from the span's
    # perpendicular axis.  A correct span always places the label ≤ ~5 ft from
    # the centreline (label is ON the beam).  The old 15% tolerance was too
    # loose: on a 1728 pt page at 3/16" scale, 15% = 259 pt ≈ 19 ft — wide
    # enough to pass a wrong-row grid fallback span (beam on row K but grid
    # snapped to row H→M, putting the midpoint 2-3 rows off).
    # 8% = ~138 pt ≈ 10 ft at 3/16" — still generous for offset labels but
    # tight enough to reject spans that land on the wrong grid row.
    PERP_TOL = 0.08
    EXT_TOL  = 0.20   # allow label up to 20 % beyond an endpoint (skewed labels)

    if beam_dir == "H":
        span_y = (by1f + by2f) / 2
        if abs(ly_frac - span_y) > PERP_TOL:
            return False
        x_lo = min(bx1f, bx2f) - EXT_TOL
        x_hi = max(bx1f, bx2f) + EXT_TOL
        return x_lo <= lx_frac <= x_hi
    elif beam_dir == "V":
        span_x = (bx1f + bx2f) / 2
        if abs(lx_frac - span_x) > PERP_TOL:
            return False
        y_lo = min(by1f, by2f) - EXT_TOL
        y_hi = max(by1f, by2f) + EXT_TOL
        return y_lo <= ly_frac <= y_hi
    else:  # "D"
        # For diagonal beams, we matched the vector line directly, so it is inherently valid.
        return True


def build_members(profiles, page_w, page_h,
                  column_symbols=None, v_grid=None, h_grid=None,
                  pts_per_foot: float = 0.0,
                  beam_dirs: dict = None,
                  beam_line_map: dict = None,
                  plan_bounds: tuple = None,
                  is_vector: bool = False):
    """
    Two-pass pipeline that prevents beam labels near a column symbol from
    being falsely promoted to "column" (the old fan-out problem).

    Pass 1 — One-to-one symbol→profile matching (greedy, closest-first).
              Each column symbol claims the single nearest unclaimed profile
              within SYMBOL_ASSOC_RADIUS.  Those profiles are definitively
              columns; no other profile can be claimed by the same symbol.

    Pass 2 — All remaining profiles are classified by TIER 2 / 3 / 4 only
              (no TIER 1 — that was handled exclusively in Pass 1).

    Position snapping is applied afterwards:
      Snap 1 — snap to nearest grid intersection.
      Snap 2 — snap to nearest unclaimed symbol (fallback).
    """
    members = []

    # ── Pass 1: exclusive symbol → profile matching ───────────────────────────
    symbol_matched_cols: set[int] = set()   # profile indices confirmed as columns
    profile_sym_pos: dict[int, tuple] = {}  # p_idx → (sx_frac, sy_frac) of matched symbol
    if column_symbols:
        # Build all (distance, symbol_idx, profile_idx) pairs within radius
        candidates = []
        for s_idx, sym in enumerate(column_symbols):
            for p_idx, p in enumerate(profiles):
                d = math.hypot(p["cx"] - sym["cx"], p["cy"] - sym["cy"])
                if d < SYMBOL_ASSOC_RADIUS:
                    candidates.append((d, s_idx, p_idx))
        candidates.sort()          # process closest pairs first
        used_s: set[int] = set()
        used_p: set[int] = set()
        for d, s_idx, p_idx in candidates:
            if s_idx not in used_s and p_idx not in used_p:
                symbol_matched_cols.add(p_idx)
                used_s.add(s_idx)
                used_p.add(p_idx)
                # Store the symbol position so the frontend region filter can
                # check it — the user draws around the SYMBOL, not the label
                profile_sym_pos[p_idx] = (
                    round(column_symbols[s_idx]["cx"] / page_w, 4),
                    round(column_symbols[s_idx]["cy"] / page_h, 4),
                )
        print(f"[BUILD] {len(symbol_matched_cols)} profiles matched to column symbols "
              f"(of {len(column_symbols)} symbols, {len(profiles)} profiles)")

    # ── Span grids: build reliable v/h grids for beam span computation ────────
    # Text-label grid detection is noisy for raster images (dense false
    # positives near the plan perimeter → tiny spans).  Column X positions are
    # far more reliable: every column sits at a real grid intersection.
    #
    # Strategy:
    #  1. Quick-classify all profiles to find columns (no symbol/grid context).
    #  2. If the text-derived v_grid is too dense (min spacing < 40 units, which
    #     is ~4 ft at 1/8" scale — physically impossible for structural steel),
    #     replace it with the sorted column X positions.
    #  3. Same for h_grid using column Y positions.
    #  4. If still empty after column fallback, use plan-boundary or page extremes
    #     so every beam at least gets a full-width/full-height line.

    pre_col_xs: list[float] = []
    pre_col_ys: list[float] = []
    for p_idx, p in enumerate(profiles):
        if p_idx in symbol_matched_cols:
            pre_col_xs.append(p["cx"])
            pre_col_ys.append(p["cy"])
        else:
            mt = classify_member(p["profile"], p["cx"], p["cy"],
                                 column_symbols=None, v_grid=None, h_grid=None)
            if mt == "column":
                pre_col_xs.append(p["cx"])
                pre_col_ys.append(p["cy"])

    def _dedup(vals: list[float], tol: float = 18) -> list[float]:
        out: list[float] = []
        for v in sorted(vals):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out

    def _grid_too_dense(grid: list, span: float, min_gap: float = 40.0) -> bool:
        """True when the grid has very closely-spaced entries (false positives)."""
        if len(grid) < 2:
            return False
        gaps = [b - a for a, b in zip(sorted(grid), sorted(grid)[1:])]
        return min(gaps) < min_gap

    # Build effective span grids
    _pb = plan_bounds or (0, 0, page_w, page_h)
    span_v_grid = v_grid or []
    span_h_grid = h_grid or []

    if _grid_too_dense(span_v_grid, page_w) or not span_v_grid:
        span_v_grid = _dedup(pre_col_xs, tol=20) if pre_col_xs else []
    if _grid_too_dense(span_h_grid, page_h) or not span_h_grid:
        span_h_grid = _dedup(pre_col_ys, tol=20) if pre_col_ys else []

    # Always include plan-boundary edges so beams beyond the last column
    # still get a line (left/right or top/bottom of the plan becomes one endpoint).
    if not span_v_grid:
        span_v_grid = [_pb[0], _pb[2]]
    else:
        span_v_grid = _dedup(sorted(span_v_grid + [_pb[0], _pb[2]]), tol=20)
    if not span_h_grid:
        span_h_grid = [_pb[1], _pb[3]]
    else:
        span_h_grid = _dedup(sorted(span_h_grid + [_pb[1], _pb[3]]), tol=20)

    print(f"[BUILD] span_v_grid({len(span_v_grid)}): {[round(x) for x in span_v_grid]}")
    print(f"[BUILD] span_h_grid({len(span_h_grid)}): {[round(y) for y in span_h_grid]}")

    # ── Pass 2: classify + snap ───────────────────────────────────────────────
    claimed_symbols: set[int] = set()

    for p_idx, p in enumerate(profiles):
        # Determine member type
        if p_idx in symbol_matched_cols:
            mtype = "column"
        else:
            # TIER 1 is handled exclusively in Pass 1 — pass column_symbols=None
            # here so classify_member uses TIER 2 / 3 / 4 only.
            mtype = classify_member(
                p["profile"], p["cx"], p["cy"],
                column_symbols=None,
                v_grid=v_grid, h_grid=h_grid,
            )

        # ── Beam-line override ────────────────────────────────────────────────
        # detect_beam_lines() matches labels that sit ON a drawn structural
        # centreline at (or near) its midpoint.  Column labels are placed AT
        # the column symbol — never at the mid-span of a long beam centreline.
        # If a profile (even one claimed by a column symbol in Pass 1) has a
        # long vector-line match, it IS a beam.  This corrects false-positive
        # column symbols (annotation boxes, beam flanges, etc.) that grab
        # nearby beam labels in Pass 1.
        _MIN_BEAM_PT = 60   # ≈ 6 ft at 1/8" — anything shorter is a tick/stub
        # Definitive column sections — square HSS and PIPE are ALWAYS columns
        # (a column viewed in plan often sits on a grid/wall line, so the line
        # match must NOT demote it to a beam).  This is what keeps HSS6X6 columns
        # on foundation / rotated sheets classified correctly.
        _pu = p["profile"].upper()
        _definitive_col = bool(re.match(r'PIPE', _pu))
        _hssm = re.match(r'HSS([\d.]+)[Xx]([\d.]+)', _pu)
        if _hssm:
            try:
                if abs(float(_hssm.group(1)) - float(_hssm.group(2))) < 0.5:
                    _definitive_col = True      # square HSS → column
            except ValueError:
                pass
        if mtype == "column" and beam_line_map and not _definitive_col:
            _hit = beam_line_map.get(p_idx)
            if _hit and _hit.get("length_pt", 0) >= _MIN_BEAM_PT:
                mtype = "beam"
                if p_idx in symbol_matched_cols:
                    symbol_matched_cols.discard(p_idx)

        # ── Section-type override ─────────────────────────────────────────────
        # Light W-sections (W10 weight<22, W12 weight<26) are NEVER used as
        # columns in structural steel framing.  If a false column symbol claimed
        # one of these labels in Pass 1 and the beam-line override didn't fire
        # (e.g. no matching vector line found), trust the section properties and
        # reclassify back to beam.
        if mtype == "column" and p_idx in symbol_matched_cols:
            _wm = re.match(r'W(\d+)[Xx](\d+)', p["profile"].upper())
            if _wm:
                _sd, _sw = int(_wm.group(1)), int(_wm.group(2))
                _clearly_beam = (
                    (_sd == 10 and _sw < 22) or
                    (_sd == 12 and _sw < 26) or
                    (_sd >= 14 and _sw < 30)
                )
                if _clearly_beam:
                    mtype = "beam"
                    symbol_matched_cols.discard(p_idx)

        render_cx, render_cy = p["cx"], p["cy"]
        snapped = False

        if mtype == "column":
            # Snap 1: snap to nearest grid intersection
            if v_grid and h_grid:
                nearest_vx = min(v_grid, key=lambda gx: abs(p["cx"] - gx))
                nearest_hy = min(h_grid, key=lambda gy: abs(p["cy"] - gy))
                if (abs(p["cx"] - nearest_vx) < GRID_SNAP_RADIUS and
                        abs(p["cy"] - nearest_hy) < GRID_SNAP_RADIUS):
                    render_cx = nearest_vx
                    render_cy = nearest_hy
                    snapped = True

            # Snap 2: snap to nearest unclaimed column symbol (fallback)
            if not snapped and column_symbols:
                best, best_dist, best_idx = None, float("inf"), -1
                for idx, sym in enumerate(column_symbols):
                    if idx in claimed_symbols:
                        continue
                    d = math.hypot(p["cx"] - sym["cx"], p["cy"] - sym["cy"])
                    if d < best_dist:
                        best_dist, best, best_idx = d, sym, idx
                if best and best_dist < SYMBOL_SNAP_RADIUS:
                    render_cx = best["cx"]
                    render_cy = best["cy"]
                    claimed_symbols.add(best_idx)
                    snapped = True  # noqa: F841

        # ── Beam direction, length, and span endpoints ────────────────────────
        beam_dir = None
        length_ft = 0.0
        bx1 = by1 = bx2 = by2 = None

        if mtype == "beam":
            line_hit = (beam_line_map or {}).get(p_idx)

            lx_frac = p["cx"] / page_w
            ly_frac = p["cy"] / page_h

            # Determine whether the vector line match is trustworthy.
            # Two rejection criteria:
            #   (A) segment too short  → dimension tick / hatch stub
            #   (B) label is far from the segment AND _span_valid fails →
            #       annotation / leader line was picked over the real beam
            #       (happens in complex plans with many non-beam lines).
            # A rejected match falls through to the grid-based fallback.
            _use_line_hit = False
            if line_hit:
                MIN_STRUCT_PT = 8    # ≈ 0.9 ft at 1/8" scale — allow short real beams
                beam_dir  = line_hit["dir"]
                length_pt = line_hit["length_pt"]

                # W-sections in framing plans are always H or V — never diagonal.
                # A diagonal match means the label was near a stair boundary, brace,
                # or section-cut line.  Reject it so grid fallback runs instead.
                _is_w_section = re.match(r'W\d+[Xx]\d+', p["profile"].upper())
                if _is_w_section and beam_dir == "D":
                    print(f"[BUILD] Rejected diagonal match for W-section "
                          f"{p['profile']} — likely stair/brace line")
                    line_hit = None

                if line_hit and length_pt < MIN_STRUCT_PT:
                    print(f"[BUILD] Vector match too short ({length_pt:.0f} pt) "
                          f"for {p['profile']} — dropped as annotation line")
                elif line_hit:
                    # ── Perpendicular-distance validation ─────────────────────
                    # Compute how far the label sits from the matched segment.
                    # Labels placed on the beam centreline are ≤ 20 pt away.
                    # Leader-line labels can be up to ~80 pt away.
                    # If the distance exceeds 80 pt, apply _span_valid as a
                    # geometric sanity check; only reject if that also fails.
                    _vx1, _vy1 = line_hit["x1"], line_hit["y1"]
                    _vx2, _vy2 = line_hit["x2"], line_hit["y2"]
                    _vdx = _vx2 - _vx1
                    _vdy = _vy2 - _vy1
                    _vln = math.hypot(_vdx, _vdy)
                    _perp_ok = True
                    if _vln > 0:
                        _tp = (((p["cx"] - _vx1) * _vdx + (p["cy"] - _vy1) * _vdy)
                               / (_vln * _vln))
                        _tc = max(0.0, min(1.0, _tp))
                        _perp_dist = math.hypot(
                            p["cx"] - (_vx1 + _tc * _vdx),
                            p["cy"] - (_vy1 + _tc * _vdy))
                        # detect_beam_lines already enforces a strict LABEL_R.
                        # Do not second-guess it unless it's outrageously far.
                        if _perp_dist > 250.0:
                            # Suspicious — run span sanity check
                            _sv_ok = _span_valid(
                                _vx1 / page_w, _vy1 / page_h,
                                _vx2 / page_w, _vy2 / page_h,
                                p["cx"] / page_w, p["cy"] / page_h,
                                beam_dir)
                            if not _sv_ok:
                                _perp_ok = False
                                print(f"[BUILD] Rejected false vector match for "
                                      f"{p['profile']} (perp={_perp_dist:.0f} pt, "
                                      f"span_valid=False) — using grid fallback")

                    if _perp_ok:
                        _use_line_hit = True
                        length_ft = (round(length_pt / pts_per_foot, 1)
                                     if pts_per_foot > 0 else 0.0)
                        bx1 = round(line_hit["x1"] / page_w, 4)
                        by1 = round(line_hit["y1"] / page_h, 4)
                        bx2 = round(line_hit["x2"] / page_w, 4)
                        by2 = round(line_hit["y2"] / page_h, 4)
                        # Chip renders at the TRUE midpoint of the span so the
                        # chip and the SVG line always coincide visually.
                        render_cx = (line_hit["x1"] + line_hit["x2"]) / 2
                        render_cy = (line_hit["y1"] + line_hit["y2"]) / 2

            if not _use_line_hit:
                # ── FALLBACK: grid-based approximation ──────────────────────
                # Runs for both vector and raster PDFs.
                # Always try BOTH H and V spans and pick the LONGER one.
                # The correct structural direction always spans the full bay
                # (longer dimension), so the wrong direction is always shorter.
                # This eliminates wrong-direction lines caused by unreliable
                # detect_beam_directions() results near openings / dense areas.
                def _try_span(direction):
                    if not (v_grid and h_grid):
                        return None
                    # Use ONLY real detected column grid positions — no plan-boundary
                    # extension.  Adding plan boundaries as synthetic columns caused
                    # two bugs: (a) very wide spans that were cleared by the 65%-plan
                    # sanity check, leaving real beams with no line; (b) stray labels
                    # near the plan edge getting full-width flying lines.
                    return compute_beam_span(
                        p["cx"], p["cy"], v_grid, h_grid, pts_per_foot, direction)

                _span_h = _try_span("H")
                _span_v = _try_span("V")
                _len_h  = _span_h["length_ft"] if _span_h else 0.0
                _len_v  = _span_v["length_ft"] if _span_v else 0.0

                # Detected direction is highly reliable; fall back to longer span if direction is missing
                _hint = ((beam_dirs or {}).get(p_idx) or p.get("dir_hint", "H"))
                if _hint == "V" and _span_v:
                    span, beam_dir = _span_v, "V"
                elif _hint == "H" and _span_h:
                    span, beam_dir = _span_h, "H"
                elif _len_h > _len_v:
                    span, beam_dir = _span_h, "H"
                elif _len_v > _len_h:
                    span, beam_dir = _span_v, "V"
                elif _hint == "V":
                    span, beam_dir = _span_v, "V"
                else:
                    span, beam_dir = _span_h, "H"

                # If primary direction failed validation, try the other.
                if span:
                    _bx1 = round(span["x1"] / page_w, 4)
                    _by1 = round(span["y1"] / page_h, 4)
                    _bx2 = round(span["x2"] / page_w, 4)
                    _by2 = round(span["y2"] / page_h, 4)
                    if not _span_valid(_bx1, _by1, _bx2, _by2, lx_frac, ly_frac, beam_dir):
                        print(f"[BUILD] Grid span invalid for {p['profile']} "
                              f"dir={beam_dir} — trying opposite direction")
                        alt_dir = "V" if beam_dir == "H" else "H"
                        alt_span = _try_span(alt_dir)
                        if alt_span:
                            ab1 = round(alt_span["x1"] / page_w, 4)
                            ab2 = round(alt_span["y1"] / page_h, 4)
                            ab3 = round(alt_span["x2"] / page_w, 4)
                            ab4 = round(alt_span["y2"] / page_h, 4)
                            if _span_valid(ab1, ab2, ab3, ab4, lx_frac, ly_frac, alt_dir):
                                span = alt_span
                                _bx1, _by1, _bx2, _by2 = ab1, ab2, ab3, ab4
                                beam_dir = alt_dir
                            else:
                                span = None   # both directions invalid
                        else:
                            span = None

                if span:
                    _bx1 = round(span["x1"] / page_w, 4)
                    _by1 = round(span["y1"] / page_h, 4)
                    _bx2 = round(span["x2"] / page_w, 4)
                    _by2 = round(span["y2"] / page_h, 4)
                    if _span_valid(_bx1, _by1, _bx2, _by2, lx_frac, ly_frac, beam_dir):
                        length_ft = span["length_ft"]
                        bx1, by1, bx2, by2 = _bx1, _by1, _bx2, _by2
                        render_cx = (span["x1"] + span["x2"]) / 2
                        render_cy = (span["y1"] + span["y2"]) / 2
                    else:
                        print(f"[BUILD] Grid span discarded for {p['profile']} "
                              f"(label=({lx_frac:.3f},{ly_frac:.3f}) "
                              f"span=({_bx1:.3f},{_by1:.3f})→({_bx2:.3f},{_by2:.3f})"
                              f" dir={beam_dir})")

        # ── Final span sanity check ──────────────────────────────────────────
        # Reject spans that exceed 65% of the plan in their primary direction.
        # Direction-aware to handle diagonal beams in rotated grids:
        #   H beam → only check width   (diagonal X-projection can be large)
        #   V beam → only check height
        #   D beam → check total length vs plan diagonal
        # This kills full-width boundary/grid lines (100%) while keeping
        # legitimate 3-of-5-bay beams (~60% plan).
        if plan_bounds and bx1 is not None and mtype == "beam":
            _pb_h_pt = plan_bounds[3] - plan_bounds[1]
            _pb_w_pt = plan_bounds[2] - plan_bounds[0]
            _span_h = abs(by2 - by1) * page_h if (by1 is not None and by2 is not None) else 0
            _span_w = abs(bx2 - bx1) * page_w if (bx1 is not None and bx2 is not None) else 0
            _too_long = False
            # Only reject spans that exceed 90% of the plan — this keeps grid/boundary
            # lines out while allowing genuine multi-bay beams (~80% of plan).
            if beam_dir == "H":
                _too_long = _span_w > _pb_w_pt * 0.90
            elif beam_dir == "V":
                _too_long = _span_h > _pb_h_pt * 0.90
            elif beam_dir == "D":
                _too_long = math.hypot(_span_w, _span_h) > math.hypot(_pb_w_pt, _pb_h_pt) * 0.90
            else:
                _too_long = (_span_h > _pb_h_pt * 0.90 or _span_w > _pb_w_pt * 0.90)
            if _too_long:
                print(f"[BUILD] Span too long for {p['profile']} dir={beam_dir} "
                      f"w={_span_w:.0f} h={_span_h:.0f} vs plan "
                      f"w={_pb_w_pt:.0f} h={_pb_h_pt:.0f} — cleared")
                bx1 = by1 = bx2 = by2 = None
                length_ft = 0.0

        # ── Structural depth-to-span sanity check ────────────────────────────
        # Research (AISC / Steel Beam Span Guide) confirms:
        #   • W12X16 (lightest W12) realistic max span = 10–14 ft normal loads
        #   • Practical L/d ratio for W-shapes ≤ 30–35 under light roof loads
        #   • Any W12 beam spanning 87 ft is physically impossible
        #
        # Rule: max_span_ft = nominal_depth_inches × 4
        #   W12 → 48 ft   W16 → 64 ft   W24 → 96 ft   W36 → 144 ft
        # This catches impossible matches (W12X16 @ 132ft from a false line/grid
        # match) while keeping all legitimate long-span girders (W36, W40, plate
        # girders).  Applies to every depth-named beam section — W, rectangular
        # HSS, and C/MC channels (a C8 channel can never span 83 ft; those are
        # always false matches to grid / border / dimension lines).  The depth is
        # the first number in the name (W16X.. / HSS16X4.. / C8X11 / MC12X31).
        if bx1 is not None and mtype == "beam" and length_ft > 0:
            _depth_match = re.match(r'(?:W|HSS|MC|C)(\d{1,2})', p["profile"], re.IGNORECASE)
            if _depth_match:
                _nom_depth = int(_depth_match.group(1))
                _max_span_ft = _nom_depth * 6     # realistic upper bound (L/d ~72)
                if length_ft > _max_span_ft:
                    print(f"[BUILD] Section-depth check failed: {p['profile']} "
                          f"span={length_ft:.1f}ft > max={_max_span_ft}ft "
                          f"(depth={_nom_depth}in × 6) — cleared")
                    bx1 = by1 = bx2 = by2 = None
                    length_ft = 0.0

        sym = profile_sym_pos.get(p_idx)
        members.append({
            "profile":   p["profile"],
            "type":      mtype,
            "length_ft": length_ft,
            "beam_dir":  beam_dir,   # "H" | "V" for beams; None for columns/braces
            # Beam span endpoints (fractions of page) for the line overlay.
            # bx1/by1 → bx2/by2 is the physical line drawn from column to column.
            "bx1": bx1, "by1": by1,
            "bx2": bx2, "by2": by2,
            # render position — midpoint of span (or snapped to grid/symbol for cols)
            "x":  round(render_cx / page_w, 4),
            "y":  round(render_cy / page_h, 4),
            # original label text position — used by region filter
            "lx": round(p["cx"] / page_w, 4),
            "ly": round(p["cy"] / page_h, 4),
            # matched column-symbol position (only for symbol-matched columns)
            "sx": sym[0] if sym else None,
            "sy": sym[1] if sym else None,
            "w":  0.025, "h": 0.012,
            "color":     MEMBER_COLORS.get(mtype, "#6B7280"),
            "confirmed": True,
            "is_column": mtype == "column",
        })

    # ── Post-processing: remove duplicates and short stubs ───────────────────
    #
    # 1. SHORT-STUB FILTER: beams under 5 ft are annotation ticks, hatch
    #    stubs, or callout leaders — never structural members.
    #
    # 2. DUPLICATE-SPAN FILTER: when multiple profile labels sit on (or very
    #    near) the same structural centreline they each independently match
    #    the same vector segment.  The result is 2-4 identical span entries
    #    for one physical beam.  We group by span endpoints (rounded to 4 pt
    #    ≈ 0.5" tolerance) and keep only the entry whose profile appears most
    #    frequently in the group (plurality vote); ties go to the heaviest
    #    section (highest weight-per-foot digit in the name).
    MIN_BEAM_FT = 1.0     # structural beams can be short framing stubs
    SPAN_TOL_PT = 8       # round endpoints to this many pts before grouping

    filtered = []
    for mem in members:
        # Always keep columns and braces
        if mem["type"] != "beam":
            filtered.append(mem)
            continue
        # Drop stubs
        if mem.get("length_ft", 0) < MIN_BEAM_FT and mem.get("bx1") is not None:
            continue
        filtered.append(mem)

    # Group beams by quantised span key
    from collections import Counter
    span_groups: dict = {}   # key → list of (index, member)
    no_span_beams = []
    for i, mem in enumerate(filtered):
        if mem["type"] != "beam" or mem.get("bx1") is None:
            no_span_beams.append(mem)
            continue
        bx1r = round(mem["bx1"] * page_w / SPAN_TOL_PT)
        by1r = round(mem["by1"] * page_h / SPAN_TOL_PT)
        bx2r = round(mem["bx2"] * page_w / SPAN_TOL_PT)
        by2r = round(mem["by2"] * page_h / SPAN_TOL_PT)
        # Normalise direction so (A→B) and (B→A) map to the same key
        key = (min(bx1r, bx2r), min(by1r, by2r), max(bx1r, bx2r), max(by1r, by2r))
        span_groups.setdefault(key, []).append(mem)

    deduped_beams = []
    for key, grp in span_groups.items():
        if len(grp) == 1:
            deduped_beams.append(grp[0])
            continue
        # Pick the profile that appears most often; break ties by weight digit
        prof_counts = Counter(m["profile"] for m in grp)
        best_count  = max(prof_counts.values())
        candidates  = [p for p, c in prof_counts.items() if c == best_count]
        def _weight_digit(prof):
            # Extract the weight number (after X) for tie-breaking: W24X68 → 68
            import re as _re
            wt = _re.search(r'[Xx](\d+)', prof)
            return int(wt.group(1)) if wt else 0
        winner = max(candidates, key=_weight_digit)
        # Keep the first member in the group that has the winning profile
        keeper = next((m for m in grp if m["profile"] == winner), grp[0])
        deduped_beams.append(keeper)
        if len(grp) > 1:
            dropped = [m["profile"] for m in grp if m is not keeper]
            print(f"[DEDUP] Merged {len(grp)} beams at same span → kept {winner}, "
                  f"dropped {dropped}")

    # ── 3. PARALLEL-OVERLAP DEDUP ─────────────────────────────────────────────
    # The exact-span grouping above only catches IDENTICAL spans.  CAD drawings
    # often draw a beam centreline twice (e.g. centreline + an adjacent flange/
    # wall-face line), producing two same-profile spans that are parallel and
    # overlapping but a few pt apart — they render as a doubled line with two
    # stacked chips.  Merge them: keep the longer span, drop the shorter.
    # Strict gate (same profile, near-parallel, perp gap ≤ 20 pt ≈ 2.2 ft,
    # real axial overlap) ensures genuine adjacent beams are never merged.
    def _seg_pt(m):
        return (m["bx1"] * page_w, m["by1"] * page_h,
                m["bx2"] * page_w, m["by2"] * page_h)

    def _is_parallel_overlap(a, b):
        ax1, ay1, ax2, ay2 = _seg_pt(a)
        bx1p, by1p, bx2p, by2p = _seg_pt(b)
        aL = math.hypot(ax2 - ax1, ay2 - ay1)
        bL = math.hypot(bx2p - bx1p, by2p - by1p)
        if aL < 1 or bL < 1:
            return False
        ux, uy = (ax2 - ax1) / aL, (ay2 - ay1) / aL
        # Parallel? (direction dot ≥ cos16° ≈ 0.96)
        if abs(((bx2p - bx1p) * ux + (by2p - by1p) * uy) / bL) < 0.96:
            return False
        # Perpendicular gap from B-midpoint to A's line
        mx, my = (bx1p + bx2p) / 2, (by1p + by2p) / 2
        perp = abs((mx - ax1) * (-uy) + (my - ay1) * ux)
        # Length ratio and axial overlap fraction (of the shorter segment)
        lr = min(aL, bL) / max(aL, bL)
        tb1 = (bx1p - ax1) * ux + (by1p - ay1) * uy
        tb2 = (bx2p - ax1) * ux + (by2p - ay1) * uy
        lo, hi = max(0.0, min(tb1, tb2)), min(aL, max(tb1, tb2))
        ovr = (hi - lo) / min(aL, bL)

        # Path 1 — DOUBLED LINE: same beam drawn a few pt apart (centreline +
        # flange/wall-face line).  Needs near-equal length and heavy overlap.
        if perp <= 20 and lr >= 0.7 and ovr >= 0.7:
            return True

        # Path 2 — COLLINEAR DUPLICATE: two same-profile spans on essentially the
        # SAME line (perp ≤ 4 pt) that overlap.  Two physically distinct beams are
        # never on the exact same centreline AND overlapping, so this is always a
        # duplicate detection — even when overlap/length differ moderately.
        # The lr ≥ 0.5 floor still protects a genuine short beam that merely runs
        # along part of a long girder (extreme length mismatch).
        if perp <= 4.0 and lr >= 0.5 and ovr >= 0.5:
            return True

        return False

    _po_kept = []
    _po_dropped = 0
    for mem in deduped_beams:
        dup_of = None
        for k in _po_kept:
            if k["profile"] == mem["profile"] and _is_parallel_overlap(k, mem):
                dup_of = k
                break
        if dup_of is None:
            _po_kept.append(mem)
        else:
            # keep whichever has the longer drawn span
            if mem.get("length_ft", 0) > dup_of.get("length_ft", 0):
                _po_kept[_po_kept.index(dup_of)] = mem
            _po_dropped += 1
    if _po_dropped:
        print(f"[DEDUP] Parallel-overlap pass removed {_po_dropped} doubled beam(s)")
    deduped_beams = _po_kept

    deduped = deduped_beams + no_span_beams + \
              [m for m in filtered if m["type"] != "beam"]
    print(f"[DEDUP] {len(members)} → {len(deduped)} members "
          f"({len(members)-len(deduped)} removed)")
    return deduped


def build_summary(members):
    s = {
        "column": 0, "beam": 0,
        "vertical_brace": 0, "horizontal_brace": 0,
        "joists": 0, "moment_connection": 0, "default_connection": 0,
        "bolt": 0, "camber": 0, "anchor": 0,
        "weld_studs": 0, "total_weight_tons": 0,
    }
    for m in members:
        t = m.get("type", "beam")
        if   t == "column":                      s["column"] += 1
        elif t == "beam":                         s["beam"]   += 1
        elif t in ("brace", "vertical_brace"):    s["vertical_brace"] += 1
        elif t == "horizontal_brace":             s["horizontal_brace"] += 1
    return s


def add_unlabeled_beam_candidates(members, all_lines, v_grid, h_grid,
                                  plan_bounds, page_w, page_h, pts_per_foot,
                                  column_symbols=None):
    """Detect unlabeled beams using DOUBLE-LINE PAIR detection.

    Every beam — labeled or not — is drawn as TWO parallel lines representing
    the top and bottom flanges of the I/W section.  Detecting a MATCHED PAIR
    of parallel lines at beam-depth spacing is a far stronger structural signal
    than any single-line heuristic:
      • dimension lines  → single line only (no matching parallel flange)
      • wall lines       → pair too close together (< W6 depth) or too far
      • curtain-wall     → boundary lines, no matching parallel at beam depth
      • stair hatch      → many closely-spaced parallels, not a clean pair
      • real beam        → exactly TWO parallel lines at section-depth spacing

    Pipeline:
      1. Bucket all solid lines into H-lines and V-lines (structural length only).
      2. For every pair within MIN_DEPTH…MAX_DEPTH spacing:
           – significant length overlap (≥ 50 % of shorter line)
           – similar lengths (within 35 %)
      3. Pair centerline = candidate beam position.
      4. Column at BOTH endpoints (augmented-grid test, shape-independent).
      5. Not already covered by a labeled beam.
      6. Not at the building edge (boundary / curtain-wall lines).
      7. Dedup — one candidate per spatial slot.
    """
    if not all_lines or not plan_bounds:
        return members

    ppf = pts_per_foot if pts_per_foot > 0 else 9.0
    pb0x, pb0y, pb1x, pb1y = plan_bounds
    plan_w, plan_h = pb1x - pb0x, pb1y - pb0y

    MIN_FT, MAX_FT = 5.0, 70.0
    GRID_TOLP = 6.0
    COL_TOL   = 40.0

    # Beam section depth at this scale (flange-to-flange pt distance).
    # W6 ≈ 6 in deep, W36 ≈ 36 in deep; convert to pts at current ppf.
    MIN_DEPTH = max(4.0,  ppf * (6.0  / 12.0))   # ~ W6  minimum
    MAX_DEPTH = min(65.0, ppf * (36.0 / 12.0))   # ~ W36 maximum
    OVERLAP_RATIO = 0.50   # flange lines must overlap ≥ 50 % of shorter line
    LEN_RATIO     = 1.35   # lengths must be within 35 % of each other

    # ── labeled-beam data ──────────────────────────────────────────────────────
    lab = [(m["bx1"]*page_w, m["by1"]*page_h,
            m["bx2"]*page_w, m["by2"]*page_h)
           for m in members
           if m.get("type") == "beam" and m.get("bx1") is not None]
    cols = [(m["x"]*page_w, m["y"]*page_h)
            for m in members if m.get("type") == "column"]
    lab_ends = []
    for (a, b, c, d) in lab:
        lab_ends += [(a, b), (c, d)]

    # ── augmented grid (covers edge zones where bubbles aren't drawn) ──────────
    def _cluster(vals, tol=9.0, minn=2):
        out, cur = [], []
        for v in sorted(vals):
            if not cur or v - cur[-1] <= tol:
                cur.append(v)
            else:
                if len(cur) >= minn:
                    out.append(sum(cur) / len(cur))
                cur = [v]
        if cur and len(cur) >= minn:
            out.append(sum(cur) / len(cur))
        return out

    # Raw symbol positions (ALL 88 detected — before emit_symbol_columns filter).
    # emit_symbol_columns drops symbols with no beam framing into them, so
    # fractional-row columns (e.g. B.2 row) and edge-zone columns that were
    # DETECTED but not EMITTED are invisible to the aug grids without this.
    # Adding them directly (no clustering — each is an exact column position)
    # closes the gap for:
    #   • Fractional grid rows  (B.2, C.6, D.1, D.2, E.4 …)
    #   • Right/left-edge columns whose beams haven't been emitted yet
    _raw_sym_x = [s["cx"] for s in (column_symbols or [])]
    _raw_sym_y = [s["cy"] for s in (column_symbols or [])]

    aug_vx = sorted(set(list(v_grid or [])
                        + _cluster([e[0] for e in lab_ends])
                        + _cluster([cx for cx, _ in cols])
                        + _raw_sym_x))
    aug_hy = sorted(set(list(h_grid or [])
                        + _cluster([e[1] for e in lab_ends])
                        + _cluster([cy for _, cy in cols])
                        + _raw_sym_y))

    def _has_col(ex, ey):
        # Primary: aug grid intersection (tight 18 pt tolerance)
        if aug_vx and aug_hy and \
           min(abs(ex-g) for g in aug_vx) < 18.0 and \
           min(abs(ey-g) for g in aug_hy) < 18.0:
            return True
        # Secondary: emitted column member position
        if any(math.hypot(ex-cx, ey-cy) < COL_TOL for cx, cy in cols):
            return True
        # Tertiary: direct raw symbol proximity — catches symbols that were
        # detected but not emitted (no beam framing yet at time of check).
        # Uses COL_TOL (40 pt) — same radius as the emitted-column check.
        if any(math.hypot(ex-s["cx"], ey-s["cy"]) < COL_TOL
               for s in (column_symbols or [])):
            return True
        return sum(1 for bx, by in lab_ends
                   if math.hypot(ex-bx, ey-by) < COL_TOL) >= 2

    def _covered(lx1, ly1, lx2, ly2):
        """True if this centerline overlaps a labeled beam (already extracted)."""
        L = math.hypot(lx2-lx1, ly2-ly1) or 1.0
        ux, uy = (lx2-lx1)/L, (ly2-ly1)/L
        for (ax1, ay1, ax2, ay2) in lab:
            aL = math.hypot(ax2-ax1, ay2-ay1) or 1.0
            if abs(((ax2-ax1)*ux + (ay2-ay1)*uy) / aL) < 0.96:
                continue
            if abs((ax1-lx1)*(-uy) + (ay1-ly1)*ux) > 18:
                continue
            t1 = (ax1-lx1)*ux + (ay1-ly1)*uy
            t2 = (ax2-lx1)*ux + (ay2-ly1)*uy
            if min(L, max(t1, t2)) - max(0.0, min(t1, t2)) > 0.40*L:
                return True
        return False

    # ── 1. bucket lines into H and V ──────────────────────────────────────────
    # H bucket: (x_start, x_end, y_mid, length, width)
    # V bucket: (x_mid, y_start, y_end, length, width)
    h_segs, v_segs = [], []
    for (x1, y1, x2, y2, ln, wv) in all_lines:
        Lft = ln / ppf
        if Lft < MIN_FT or Lft > MAX_FT:
            continue
        adx, ady = abs(x2-x1), abs(y2-y1)
        if adx > ady * 2:                          # horizontal
            if adx > plan_w * 0.85: continue       # full-width border → skip
            if x1 > x2: x1, y1, x2, y2 = x2, y2, x1, y1
            ym = (y1+y2) / 2
            if h_grid and min(abs(ym-g) for g in h_grid) < GRID_TOLP:
                continue                            # ON a grid row → skip
            h_segs.append((x1, x2, ym, ln, wv))
        elif ady > adx * 2:                        # vertical
            if ady > plan_h * 0.85: continue
            if y1 > y2: x1, y1, x2, y2 = x2, y2, x1, y1
            xm = (x1+x2) / 2
            if v_grid and min(abs(xm-g) for g in v_grid) < GRID_TOLP:
                continue
            v_segs.append((xm, y1, y2, ln, wv))

    # ── 2. find matched flange pairs ──────────────────────────────────────────
    raw_pairs = []   # (cx1, cy1, cx2, cy2, cln, bdir)

    # Horizontal pairs — sort by y_mid, scan upward within MAX_DEPTH
    h_by_y = sorted(h_segs, key=lambda s: s[2])
    for i, (axs, axe, ay, aln, _aw) in enumerate(h_by_y):
        for j in range(i+1, len(h_by_y)):
            bxs, bxe, by, bln, _bw = h_by_y[j]
            dy = by - ay
            if dy > MAX_DEPTH: break
            if dy < MIN_DEPTH: continue
            # At least one flange must be a real structural line.
            # Annotation/dimension/hatch lines are hairline (< 0.25 pt);
            # beam flanges are always >= 0.3 pt.  Two hairlines = false pair.
            if max(_aw, _bw) < 0.3: continue
            ovlp = min(axe, bxe) - max(axs, bxs)
            if ovlp < OVERLAP_RATIO * min(aln, bln): continue
            if max(aln, bln) / (min(aln, bln) or 1) > LEN_RATIO: continue
            cy  = (ay + by) / 2
            cx1 = max(axs, bxs);  cx2 = min(axe, bxe)
            cln = cx2 - cx1
            if cln / ppf < MIN_FT: continue
            raw_pairs.append((cx1, cy, cx2, cy, cln, "H"))

    # Vertical pairs — sort by x_mid, scan rightward within MAX_DEPTH
    v_by_x = sorted(v_segs, key=lambda s: s[0])
    for i, (ax, ays, aye, aln, _aw) in enumerate(v_by_x):
        for j in range(i+1, len(v_by_x)):
            bx, bys, bye, bln, _bw = v_by_x[j]
            dx = bx - ax
            if dx > MAX_DEPTH: break
            if dx < MIN_DEPTH: continue
            if max(_aw, _bw) < 0.3: continue   # hairline filter — same as H pairs
            ovlp = min(aye, bye) - max(ays, bys)
            if ovlp < OVERLAP_RATIO * min(aln, bln): continue
            if max(aln, bln) / (min(aln, bln) or 1) > LEN_RATIO: continue
            cx  = (ax + bx) / 2
            cy1 = max(ays, bys);  cy2 = min(aye, bye)
            cln = cy2 - cy1
            if cln / ppf < MIN_FT: continue
            raw_pairs.append((cx, cy1, cx, cy2, cln, "V"))

    # ── 3. filter and emit ────────────────────────────────────────────────────
    EDGE_M = COL_TOL + 8.0   # 48 pt — wider than col check so edge lines are caught
    seen, n = [], 0

    for (cx1, cy1, cx2, cy2, cln, bdir) in raw_pairs:
        mx, my = (cx1+cx2)/2, (cy1+cy2)/2
        Lft = cln / ppf

        # ── Column check FIRST ────────────────────────────────────────────────
        # A real structural beam always connects columns at BOTH endpoints.
        # Check this before edge rejection — a beam in the rightmost/leftmost
        # bay has its midpoint near the plan edge, so the old edge-first order
        # was killing real perimeter beams (circled area bug).
        # Only apply edge rejection to pairs with NO column confirmation — those
        # are curtain-wall traces, CMU boundary lines, or full-height facade
        # lines that happen to form a pair but connect to nothing structural.
        has_both_cols = _has_col(cx1, cy1) and _has_col(cx2, cy2)

        if not has_both_cols:
            # No confirmed columns → apply edge guard to keep out boundary /
            # facade lines.  With confirmed columns the beam passes even at edge.
            if bdir == "V" and aug_vx:
                vlo, vhi = min(aug_vx), max(aug_vx)
                if abs(mx-vlo) < EDGE_M or abs(mx-vhi) < EDGE_M:
                    continue
            if bdir == "H" and aug_hy:
                hlo, hhi = min(aug_hy), max(aug_hy)
                if abs(my-hlo) < EDGE_M or abs(my-hhi) < EDGE_M:
                    continue
            # No columns and not caught by edge guard → still not a beam
            continue

        # not already a labeled beam
        if _covered(cx1, cy1, cx2, cy2):
            continue

        # dedup
        if any(math.hypot(mx-sx, my-sy) < 30 for sx, sy in seen):
            continue
        seen.append((mx, my))

        members.append({
            "profile": "(beam?)", "type": "beam",
            "length_ft": round(Lft, 1),
            "beam_dir": bdir,
            "bx1": round(cx1/page_w, 4), "by1": round(cy1/page_h, 4),
            "bx2": round(cx2/page_w, 4), "by2": round(cy2/page_h, 4),
            "x":  round(mx/page_w, 4),   "y":  round(my/page_h, 4),
            "lx": round(mx/page_w, 4),   "ly": round(my/page_h, 4),
            "sx": None, "sy": None, "w": 0.025, "h": 0.012,
            "color": "#F59E0B",
            "confirmed": False, "is_column": False,
            "unlabeled": True, "confidence": "low",
        })
        n += 1

    if n:
        print(f"[UNLABELED] {n} unlabeled beam candidates (double-line pair)")

    # ── ADAPTIVE FALLBACK: single thick-centerline style ─────────────────────
    # Some CAD exports draw each beam as ONE wide stroke (not two flange lines).
    # In that case raw_pairs is empty even on a drawing with many unlabeled beams.
    # Auto-detect: if we found 0 pairs AND labeled beams exist, the drawing uses
    # single-line style → run the single-line detector with the proven filters.
    # ── auto-detect drawing style ─────────────────────────────────────────────
    # In a DOUBLE-LINE drawing, beam flanges are drawn thin (≤1.5 pt each).
    # In a SINGLE-LINE drawing, each beam is ONE thick stroke (>1.5 pt).
    # Check the max stroke weight among structural-length H lines to decide.
    _max_w = max((wv for _,_,_,_,wv in h_segs), default=0.0)
    _is_single_line = _max_w > 1.5
    print(f"[UNLABELED] style={'single-line(fallback)' if _is_single_line else 'double-line(pairs)'}  max_h_w={_max_w:.1f}pt")
    if _is_single_line and lab:
        THICK      = 0.5          # minimum stroke width for a structural line
        MIN_FT_SL  = 6.5          # raise floor: 5-6ft = wall pocket / stair trim, not framing beams
        EDGE_M2    = max(COL_TOL * 1.5, 55.0)   # wider V-edge filter (curtain wall may be slightly off max aug_vx)
        H_EDGE_TOL = GRID_TOLP * 1.5             # tight H-edge: only exact top/bottom row lines
        for (x1, y1, x2, y2, ln, wv) in all_lines:
            if wv < THICK: continue
            Lft = ln / ppf
            if Lft < MIN_FT_SL or Lft > MAX_FT: continue
            adx, ady = abs(x2-x1), abs(y2-y1)
            if adx > ady * 2:
                bdir = "H"
                if adx > plan_w * 0.85: continue
            elif ady > adx * 2:
                bdir = "V"
                if ady > plan_h * 0.85: continue
            else:
                continue
            my = (y1+y2)/2;  mx = (x1+x2)/2
            if bdir == "H" and h_grid and min(abs(my-g) for g in h_grid) < GRID_TOLP:
                continue
            if bdir == "V" and v_grid and min(abs(mx-g) for g in v_grid) < GRID_TOLP:
                continue
            # V-edge structural rule: an INTERIOR unlabeled beam has labeled
            # horizontal beams framing into it from BOTH sides (left and right).
            # A curtain-wall / boundary line only has beams from ONE side.
            # This is drawing-independent — no tolerance tuning needed.
            if bdir == "V":
                vy_lo, vy_hi = min(y1, y2), max(y1, y2)
                left_cnt  = sum(1 for ax1,ay1,ax2,ay2 in lab
                                if abs(min(ax1,ax2)-mx) < COL_TOL
                                and vy_lo-COL_TOL < (ay1+ay2)/2 < vy_hi+COL_TOL)
                right_cnt = sum(1 for ax1,ay1,ax2,ay2 in lab
                                if abs(max(ax1,ax2)-mx) < COL_TOL
                                and vy_lo-COL_TOL < (ay1+ay2)/2 < vy_hi+COL_TOL)
                if left_cnt == 0 or right_cnt == 0:
                    continue   # beams only from one side → boundary, not interior beam
            # H-edge: lines exactly ON the outermost top/bottom row are border lines
            # (use tight tolerance so interior rows at top/bottom are still kept)
            if bdir == "H" and aug_hy:
                hlo, hhi = min(aug_hy), max(aug_hy)
                if abs(my-hlo) < H_EDGE_TOL or abs(my-hhi) < H_EDGE_TOL: continue
            if not (_has_col(x1, y1) and _has_col(x2, y2)): continue
            if _covered(x1, y1, x2, y2): continue
            # No hatch filter here — column-at-both-ends is sufficient in
            # single-line mode.  The hatch filter would reject real beams near
            # other labeled beams (they appear as parallel lines within 28 pt).
            if any(math.hypot(mx-sx, my-sy) < 30 for sx, sy in seen): continue
            seen.append((mx, my))
            members.append({
                "profile": "(beam?)", "type": "beam",
                "length_ft": round(Lft, 1), "beam_dir": bdir,
                "bx1": round(x1/page_w,4), "by1": round(y1/page_h,4),
                "bx2": round(x2/page_w,4), "by2": round(y2/page_h,4),
                "x": round(mx/page_w,4),   "y": round(my/page_h,4),
                "lx": round(mx/page_w,4),  "ly": round(my/page_h,4),
                "sx": None, "sy": None, "w": 0.025, "h": 0.012,
                "color": "#F59E0B",
                "confirmed": False, "is_column": False,
                "unlabeled": True, "confidence": "low",
            })
            n += 1
        if n:
            print(f"[UNLABELED] {n} unlabeled beam candidates (single-line fallback)")

    return members


def add_unlabeled_lines_universal(members, all_struct_lns, plan_bounds,
                                  page_w, page_h, pts_per_foot,
                                  v_grid=None, h_grid=None,
                                  column_symbols=None):
    """UNIVERSAL unlabeled-beam detection — works at ANY label count.

    Every structural beam line that is NOT already claimed by a labeled beam
    becomes a (beam?) candidate.  Strict false-positive filters ensure only
    genuine structural framing members are flagged — dimension lines, column
    stubs, wall lines, and grid lines are rejected.

    Filters applied (in order):
      F1  Orthogonal only (H / V) — diagonals are braces, not beams.
      F2  Length < 75 % of plan width (H) or height (V) — full-span = grid line.
      F3  Minimum span ≥ 5 ft — stubs / connection plates are shorter.
      F4  Both endpoints must land at a real column position (grid intersection
          or detected column symbol).  This single gate eliminates:
            • dimension lines  (endpoints at annotation ticks, not columns)
            • wall / CMU lines (endpoints at walls, not grid intersections)
            • column stub lines (both ends inside the same column footprint)
          When no grid/symbol data exists the check is skipped (safe fallback).
      F5  Midpoint dedup — one candidate per 40-pt slot.
      F6  Coverage — skip if a labeled beam already occupies this centreline.
      F7  Hard cap 200.
    """
    if not all_struct_lns or not plan_bounds:
        return members
    ppf = pts_per_foot if pts_per_foot > 0 else 9.0
    pb0x, pb0y, pb1x, pb1y = plan_bounds
    plan_w, plan_h = pb1x - pb0x, pb1y - pb0y
    MAX_H_FRAC = 0.75   # tightened from 0.85 — long dimension lines span ~80 %
    MAX_V_FRAC = 0.75
    DEDUP_R    = 40.0
    CAP        = 200
    MIN_FT     = 5.0    # F3: beams shorter than 5 ft are stubs / conn. plates

    # ── F4: column-endpoint check ─────────────────────────────────────────────
    # Build augmented column X / Y position lists from grid lines + symbols.
    # Grid lines mark column centrelines; symbols mark individual columns.
    _col_xs = sorted(set(
        [float(x) for x in (v_grid or [])] +
        [s["cx"] for s in (column_symbols or [])]
    ))
    _col_ys = sorted(set(
        [float(y) for y in (h_grid or [])] +
        [s["cy"] for s in (column_symbols or [])]
    ))
    _has_col_data = bool(_col_xs and _col_ys)

    # Tolerance: 36 pt ≈ 4 ft at 1/8" scale.  Generous enough to bridge a
    # drawn-endpoint → column-centre gap without crossing to the next bay
    # (typical minimum steel bay is ~8 ft, so 4 ft is well inside one bay).
    COL_TOL = max(36.0, ppf * 0.5)

    def _at_col(ex, ey):
        """True if (ex, ey) is at a column position (grid intersection or symbol)."""
        if not _has_col_data:
            return True   # no grid data → skip check (safe: no false rejections)
        return (min(abs(ex - x) for x in _col_xs) < COL_TOL and
                min(abs(ey - y) for y in _col_ys) < COL_TOL)

    # ── Coverage check against already-extracted labeled beams ────────────────
    # Exclude any prior unlabeled candidates so coverage is measured only
    # against real labels — prevents cascading self-suppression.
    lab = [(m["bx1"] * page_w, m["by1"] * page_h,
            m["bx2"] * page_w, m["by2"] * page_h)
           for m in members
           if m.get("type") == "beam" and m.get("bx1") is not None
           and not m.get("unlabeled")]

    def _covered(lx1, ly1, lx2, ly2):
        """True if this line coincides with a labeled beam (already extracted)."""
        L = math.hypot(lx2 - lx1, ly2 - ly1) or 1.0
        ux, uy = (lx2 - lx1) / L, (ly2 - ly1) / L
        for (ax1, ay1, ax2, ay2) in lab:
            aL = math.hypot(ax2 - ax1, ay2 - ay1) or 1.0
            if abs(((ax2 - ax1) * ux + (ay2 - ay1) * uy) / aL) < 0.96:
                continue                      # not parallel
            if abs((ax1 - lx1) * (-uy) + (ay1 - ly1) * ux) > 18:
                continue                      # too far perpendicular
            t1 = (ax1 - lx1) * ux + (ay1 - ly1) * uy
            t2 = (ax2 - lx1) * ux + (ay2 - ly1) * uy
            if min(L, max(t1, t2)) - max(0.0, min(t1, t2)) > 0.4 * L:
                return True                   # overlaps the labeled beam
        return False

    seen, n = [], 0
    rejected_col, rejected_len, rejected_dir = 0, 0, 0
    for (lx1, ly1, lx2, ly2, ln) in all_struct_lns:
        if n >= CAP:
            break
        adx, ady = abs(lx2 - lx1), abs(ly2 - ly1)

        # F1: orthogonal only + F2: reject near-full-plan-width/height lines
        if adx > ady * 2:
            bdir = "H"
            if adx > plan_w * MAX_H_FRAC:
                rejected_dir += 1
                continue
        elif ady > adx * 2:
            bdir = "V"
            if ady > plan_h * MAX_V_FRAC:
                rejected_dir += 1
                continue
        else:
            rejected_dir += 1
            continue                          # diagonal → skip

        # F3: minimum structural length
        if ln / ppf < MIN_FT:
            rejected_len += 1
            continue

        # F4: BOTH endpoints must be at a real column position.
        # This is the primary false-positive gate — dimension lines, wall lines,
        # and grid lines do NOT span column-to-column in both axes.
        if not (_at_col(lx1, ly1) and _at_col(lx2, ly2)):
            rejected_col += 1
            continue

        # ── Column-centreline snap ────────────────────────────────────────────
        # Beams are drawn column-FACE to column-FACE in CAD (the clear span),
        # but we display column-CENTRE to column-CENTRE (the overall span).
        # Snap each endpoint to the nearest column grid position — identical to
        # Pass 3d in detect_beam_lines — so unlabelled lines reach exactly
        # column-to-column, matching labelled beam behaviour.
        # Only snap when the two ends target DIFFERENT column positions so we
        # never collapse both endpoints onto the same column.
        if _has_col_data:
            if bdir == "H" and _col_xs:
                _gx1 = min(_col_xs, key=lambda gx: abs(lx1 - gx))
                _gx2 = min(_col_xs, key=lambda gx: abs(lx2 - gx))
                if abs(_gx1 - _gx2) > 5:
                    if abs(lx1 - _gx1) <= COL_TOL:
                        lx1 = _gx1
                    if abs(lx2 - _gx2) <= COL_TOL:
                        lx2 = _gx2
            elif bdir == "V" and _col_ys:
                _gy1 = min(_col_ys, key=lambda gy: abs(ly1 - gy))
                _gy2 = min(_col_ys, key=lambda gy: abs(ly2 - gy))
                if abs(_gy1 - _gy2) > 5:
                    if abs(ly1 - _gy1) <= COL_TOL:
                        ly1 = _gy1
                    if abs(ly2 - _gy2) <= COL_TOL:
                        ly2 = _gy2

        # Recompute midpoint + length after snap
        mx, my = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        ln = math.hypot(lx2 - lx1, ly2 - ly1)

        # F5: midpoint dedup
        if any(math.hypot(mx - sx, my - sy) < DEDUP_R for sx, sy in seen):
            continue

        # F6: coverage — skip lines already claimed by a labeled beam
        if _covered(lx1, ly1, lx2, ly2):
            continue

        seen.append((mx, my))
        Lft = ln / ppf
        members.append({
            "profile": "(beam?)", "type": "beam",
            "length_ft": round(Lft, 1),
            "beam_dir": bdir,
            "bx1": round(lx1 / page_w, 4), "by1": round(ly1 / page_h, 4),
            "bx2": round(lx2 / page_w, 4), "by2": round(ly2 / page_h, 4),
            "x":  round(mx / page_w, 4),   "y":  round(my / page_h, 4),
            "lx": round(mx / page_w, 4),   "ly": round(my / page_h, 4),
            "sx": None, "sy": None, "w": 0.025, "h": 0.012,
            "color": "#F59E0B",
            "confirmed": False, "is_column": False,
            "unlabeled": True, "confidence": "low",
        })
        n += 1

    print(f"[UNLABELED] {n} universal (beam?) candidates "
          f"(rejected: dir/len={rejected_dir+rejected_len} col_gate={rejected_col})")
    return members


def emit_symbol_columns(members, column_symbols, v_grid, h_grid,
                        page_w, page_h):
    """Emit a COLUMN member for each detected column symbol that sits at a grid
    intersection and isn't already represented by a column member.

    On framing plans the columns are I/H symbols at grid crossings with no size
    label of their own (they're sized on the Column Schedule), so they were
    never counted.  Here we count them: position = symbol, size left blank (a
    later Column-Schedule pass can fill it).  Gating on a grid intersection
    (near a vertical AND a horizontal grid line) keeps false marks out.
    """
    if not column_symbols:
        return members
    FRAME = 48.0         # a beam endpoint this close ⇒ a beam frames into it
    SNAP  = 30.0         # snap marker to grid line if within this (precise centre)
    existing = [(m["x"] * page_w, m["y"] * page_h)
                for m in members if m.get("type") == "column"]
    # Beam endpoints — a real column is WHERE BEAMS MEET; a stray text / CMU /
    # dimension / label mark has no beam framing into it (its nearest beam end is
    # a full bay away).  This is the reliable filter that rejects false symbols.
    beam_ends = []
    for m in members:
        if m.get("type") == "beam" and m.get("bx1") is not None:
            beam_ends.append((m["bx1"] * page_w, m["by1"] * page_h))
            beam_ends.append((m["bx2"] * page_w, m["by2"] * page_h))
    # A real column is WHERE BEAMS MEET.  Requiring a beam to frame into the
    # symbol guarantees every COL marker is a genuine column (no stray text /
    # CMU / dimension / base-plate mark).  (Foundation sheets with no framing
    # beams will undercount until they get a dedicated mode — but they will show
    # ZERO false columns, which is the requirement.)
    added = []
    for s in column_symbols:
        cx, cy = s["cx"], s["cy"]
        # FILTER: a beam must frame into this symbol — else it is not a column.
        if not any(math.hypot(cx - ex, cy - ey) < FRAME for ex, ey in beam_ends):
            continue
        # PLACE at the column centre: snap to the nearby grid line(s) when close
        # (precise), otherwise keep the symbol's own centre.
        gx = min(v_grid, key=lambda g: abs(cx - g)) if v_grid else cx
        gy = min(h_grid, key=lambda g: abs(cy - g)) if h_grid else cy
        px = gx if abs(cx - gx) <= SNAP else cx
        py = gy if abs(cy - gy) <= SNAP else cy
        # not already a column here
        if any(math.hypot(px - ex, py - ey) < 40 for ex, ey in existing + added):
            continue
        added.append((px, py))
        members.append({
            "profile": "COL", "type": "column", "length_ft": 0.0,
            "beam_dir": None,
            "bx1": None, "by1": None, "bx2": None, "by2": None,
            "x": round(px / page_w, 4), "y": round(py / page_h, 4),
            "lx": round(px / page_w, 4), "ly": round(py / page_h, 4),
            "sx": round(px / page_w, 4), "sy": round(py / page_h, 4),
            "w": 0.018, "h": 0.018,
            "color": MEMBER_COLORS["column"], "confirmed": True,
            "is_column": True, "size_unknown": True,
        })
    if added:
        print(f"[COLUMNS] emitted {len(added)} symbol columns at grid intersections")
    return members


def clean_column_lines(v_grid, h_grid, column_symbols, tol=8.0):
    """Build a CLEAN column-centre set (X for vertical lines, Y for horizontal).

    Per the special rule "columns repeat at grid intersections": a real column
    line carries MULTIPLE column symbols at the same X (or Y).  So we keep the
    grid lines PLUS only the symbol positions where ≥2 symbols align — scattered
    single false marks (connection details mistaken for columns) are dropped —
    then merge anything within `tol`.  This removes the false closely-spaced
    clusters that made the column-aware logic leak.
    """
    def _aligned(vals, min_n=2):
        out, cur = [], []
        for v in sorted(vals):
            if not cur or v - cur[-1] <= tol:
                cur.append(v)
            else:
                if len(cur) >= min_n:
                    out.append(sum(cur) / len(cur))
                cur = [v]
        if cur and len(cur) >= min_n:
            out.append(sum(cur) / len(cur))
        return out

    def _merge(vals, m=8.0):
        out = []
        for v in sorted(vals):
            if not out or abs(v - out[-1]) > m:
                out.append(v)
            else:
                out[-1] = (out[-1] + v) / 2.0
        return out

    sx = _aligned([s["cx"] for s in (column_symbols or [])])
    sy = _aligned([s["cy"] for s in (column_symbols or [])])
    cx = _merge([float(g) for g in (v_grid or [])] + sx)
    cy = _merge([float(g) for g in (h_grid or [])] + sy)
    return cx, cy


def trim_beam_overshoot(members, page_w, page_h, pts_per_foot,
                        col_x=None, col_y=None):
    """Trim a beam endpoint back to the perpendicular beam it frames into.

    A secondary beam frames into a girder (a perpendicular beam) and STOPS at
    its side — it must not pass through it.  For each beam endpoint, if a roughly
    perpendicular beam crosses the beam's axis just INSIDE the endpoint (so the
    beam overshoots that crossing by a small amount), pull the endpoint back to
    the crossing.  Only ever SHORTENS a beam, never lengthens it.

    COLUMN-AWARE: an endpoint sitting at a real COLUMN (from the CLEAN column set)
    is left alone — the beam reaches column-centre to column-centre there, even
    if a girder crosses just before the column.  Only girder-overshoots with NO
    column at the end are trimmed.
    """
    ppf = pts_per_foot if pts_per_foot > 0 else 9.0
    MAX_TRIM = 6.0 * ppf      # only trim overshoots up to ~6 ft (not real spans)
    MIN_TRIM = 0.7 * ppf      # ignore <0.7 ft (that's a normal connection gap)
    PERP_DOT = 0.5            # |cos| < 0.5  → >60° apart  → perpendicular-ish
    COL_ZONE = 1.8 * ppf      # endpoint within this of a column ⇒ it's AT a column

    _cx = sorted(col_x or [])
    _cy = sorted(col_y or [])

    def _at_col(v, cols):
        return bool(cols) and min(abs(v - c) for c in cols) < COL_ZONE

    segs = []
    for m in members:
        if m.get("type") == "beam" and m.get("bx1") is not None:
            segs.append((m, m["bx1"] * page_w, m["by1"] * page_h,
                         m["bx2"] * page_w, m["by2"] * page_h))

    # Outermost column-line bounds: a beam must not extend beyond the last
    # detected column line.  This catches CMU-wall pockets and curtain-wall
    # edges where no perpendicular beam exists to trigger the main trim.
    _cx_min = min(_cx) if _cx else None
    _cx_max = max(_cx) if _cx else None
    _cy_min = min(_cy) if _cy else None
    _cy_max = max(_cy) if _cy else None
    EDGE_TOL = COL_ZONE * 0.5            # tightened: only tiny cantilever gap allowed
    _n_b2b = 0                          # diagnostic: count of beams trimmed

    for m, ax1, ay1, ax2, ay2 in segs:
        L = math.hypot(ax2 - ax1, ay2 - ay1)
        if L < 1:
            continue
        ux, uy = (ax2 - ax1) / L, (ay2 - ay1) / L
        _is_h = abs(ux) >= abs(uy)
        _A_at_col = _at_col(ax1 if _is_h else ay1, _cx if _is_h else _cy)
        _B_at_col = _at_col(ax2 if _is_h else ay2, _cx if _is_h else _cy)
        new_t0, new_tL = 0.0, L

        # ── beam-to-beam trim (original logic) ───────────────────────────────
        for (n, bx1, by1, bx2, by2) in segs:
            if n is m:
                continue
            bL = math.hypot(bx2 - bx1, by2 - by1)
            if bL < 1:
                continue
            vx, vy = (bx2 - bx1) / bL, (by2 - by1) / bL
            if abs(ux * vx + uy * vy) > PERP_DOT:
                continue                       # not perpendicular enough
            den = ux * vy - uy * vx
            if abs(den) < 1e-6:
                continue
            t = ((bx1 - ax1) * vy - (by1 - ay1) * vx) / den
            u = ((bx1 - ax1) * uy - (by1 - ay1) * ux) / den
            if not (-6.0 <= u <= bL + 6.0):
                continue                       # crossing not on the perpendicular beam
            if (not _B_at_col) and MIN_TRIM < (L - t) <= MAX_TRIM and t > L * 0.5:
                new_tL = max(min(new_tL, t), L - MAX_TRIM)
            if (not _A_at_col) and MIN_TRIM < t <= MAX_TRIM and t < L * 0.5:
                new_t0 = min(max(new_t0, t), MAX_TRIM)

        # ── edge-boundary clamp: clamp to outermost column line ───────────────
        # Beyond the OUTERMOST detected column line there is provably no structure
        # to frame into, so ANY endpoint protruding past it (more than EDGE_TOL)
        # is pure overshoot — clamp it straight back to that line with NO MAX_TRIM
        # cap.  This is what kills the perimeter overshoot (corner/edge beams that
        # stick out >6 ft past the last column, which the girder trim's 6 ft cap
        # let through).  The girder trim above keeps its cap because a long real
        # span CAN exist between two interior girders; out here it cannot.  A real
        # cantilever overhang stays within EDGE_TOL and is left alone.
        if _is_h:
            if _cx_min is not None and ax1 < _cx_min - EDGE_TOL and not _A_at_col:
                t_edge = _cx_min - ax1
                if t_edge > MIN_TRIM:
                    new_t0 = max(new_t0, t_edge)
            if _cx_max is not None and ax2 > _cx_max + EDGE_TOL and not _B_at_col:
                t_edge = L - (_cx_max - ax1)
                if t_edge > MIN_TRIM:
                    new_tL = min(new_tL, L - t_edge)
        else:
            if _cy_min is not None and ay1 < _cy_min - EDGE_TOL and not _A_at_col:
                t_edge = _cy_min - ay1
                if t_edge > MIN_TRIM:
                    new_t0 = max(new_t0, t_edge)
            if _cy_max is not None and ay2 > _cy_max + EDGE_TOL and not _B_at_col:
                t_edge = L - (_cy_max - ay1)
                if t_edge > MIN_TRIM:
                    new_tL = min(new_tL, L - t_edge)
        # Safety: never let the combined trims collapse a beam to a sliver.  If
        # they would leave less than a real minimum span (3 ft), the outermost-
        # column estimate is more likely wrong than the beam — leave it untouched.
        MIN_SPAN = 3.0 * ppf
        if (new_t0 > 0 or new_tL < L) and (new_tL - new_t0) >= MIN_SPAN:
            nx1, ny1 = ax1 + ux * new_t0, ay1 + uy * new_t0
            nx2, ny2 = ax1 + ux * new_tL, ay1 + uy * new_tL
            m["bx1"] = round(nx1 / page_w, 4); m["by1"] = round(ny1 / page_h, 4)
            m["bx2"] = round(nx2 / page_w, 4); m["by2"] = round(ny2 / page_h, 4)
            m["x"] = round((nx1 + nx2) / 2 / page_w, 4)
            m["y"] = round((ny1 + ny2) / 2 / page_h, 4)
            m["length_ft"] = round(math.hypot(nx2 - nx1, ny2 - ny1) / ppf, 1)
            _n_b2b += 1
    print(f"[TRIM] columns: {len(_cx)} X / {len(_cy)} Y  |  beams trimmed: {_n_b2b}")
    return members


def snap_beam_ends_to_supports(members, all_struct_lns, col_x, col_y,
                               page_w, page_h, pts_per_foot):
    """UNIVERSAL endpoint rule: a beam runs SUPPORT-to-SUPPORT.

    Real framing data shows ~70 % of beams have NO column on their body — they
    frame girder-to-girder or girder-to-column.  So an endpoint must terminate
    at the nearest PERPENDICULAR structural line it reaches — a column grid line
    OR a crossing girder (a perpendicular drawn beam line) — and STOP there.

    For each endpoint we look at every perpendicular support crossing within a
    search window and snap to the one nearest that endpoint.  This both TRIMS
    overshoot (endpoint sits past the support → pull in) and CLOSES short gaps
    (endpoint stops short of the support → push out) — symmetric, exactly how a
    correctly-drawn labeled girder already terminates.  This is what makes the
    overlay "touch the perpendicular line and stop" on every beam, labeled or not.

    Safety: never collapses a beam below MIN_SPAN; only moves an endpoint when a
    genuine perpendicular support exists within the window (otherwise left as-is).
    """
    if not members:
        return members
    ppf      = pts_per_foot if pts_per_foot > 0 else 9.0
    # Asymmetric window.  Trimming INWARD is safe to do generously: when a beam
    # overshoots its support the gap beyond the support is empty, so the nearest
    # crossing found inward IS the true support even for big overshoots.  EXTENDING
    # outward is riskier (could grab a far line), so keep it short.
    IN_WIN   = 14.0 * ppf     # trim overshoot up to ~14 ft inward to a support
    OUT_WIN  = 4.0  * ppf     # extend a short end only up to ~4 ft to a support
    MIN_SPAN = 3.0  * ppf     # never trim a beam shorter than a real minimum span
    PERP_DOT = 0.30           # |cos| < 0.30  → >72°  → perpendicular-ish
    cols_x   = sorted(col_x or [])
    cols_y   = sorted(col_y or [])

    for m in members:
        if m.get("type") != "beam" or m.get("bx1") is None:
            continue
        x1 = m["bx1"] * page_w; y1 = m["by1"] * page_h
        x2 = m["bx2"] * page_w; y2 = m["by2"] * page_h
        L  = math.hypot(x2 - x1, y2 - y1)
        if L < 1:
            continue
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        is_h   = abs(ux) >= abs(uy)

        cross = []   # parametric t along the axis (from endpoint A) of each support crossing

        # 1) Column grid lines perpendicular to the beam (centre-to-centre spans).
        if is_h and abs(ux) > 1e-6:
            cross += [(cx - x1) / ux for cx in cols_x]
        elif (not is_h) and abs(uy) > 1e-6:
            cross += [(cy - y1) / uy for cy in cols_y]

        # 2) Perpendicular drawn structural lines (girders) — the support for the
        #    ~70 % of beams that DON'T land on a column.  Crossing must fall on the
        #    girder's actual body (not its infinite extension).
        for (sx1, sy1, sx2, sy2, _l) in (all_struct_lns or []):
            sdx, sdy = sx2 - sx1, sy2 - sy1
            sl = math.hypot(sdx, sdy)
            if sl < 1:
                continue
            vx, vy = sdx / sl, sdy / sl
            if abs(ux * vx + uy * vy) > PERP_DOT:
                continue                                   # not perpendicular
            den = ux * vy - uy * vx
            if abs(den) < 1e-6:
                continue
            t = ((sx1 - x1) * vy - (sy1 - y1) * vx) / den  # param along beam
            u = ((sx1 - x1) * uy - (sy1 - y1) * ux) / den  # param along girder
            if -6.0 <= u <= sl + 6.0:                      # crossing on girder body
                cross.append(t)

        # endpoint A (t=0): trim inward (t>0) up to IN_WIN, extend outward (t<0)
        # up to OUT_WIN; endpoint B (t=L): trim inward (t<L) up to IN_WIN, extend
        # outward (t>L) up to OUT_WIN.  Pick the support crossing nearest the end.
        near0 = [t for t in cross if -OUT_WIN <= t <= IN_WIN] if cross else []
        near1 = [t for t in cross if L - IN_WIN <= t <= L + OUT_WIN] if cross else []
        new0 = min(near0, key=lambda t: abs(t))       if near0 else 0.0
        new1 = min(near1, key=lambda t: abs(t - L))   if near1 else L

        # ── Drawn-steel clamp ────────────────────────────────────────────────
        # The overlay must not extend more than half a support-depth beyond the
        # ACTUAL drawn steel line it traces.  build_members extends each end to the
        # nearest COLUMN (≤ ~3.7 ft), but ~70 % of beams frame into a GIRDER that
        # sits closer — so that extension overshoots PAST the girder to a column
        # beyond it.  Clamping to the drawn line + PAD pulls the end back onto the
        # girder it actually touches.  Collinear drawn segments (CAD breaks the
        # centreline at each girder crossing) are chained into one extent.
        PAD = 1.5 * ppf
        px, py = -uy, ux
        d_ts = []
        for (sx1, sy1, sx2, sy2, _l) in (all_struct_lns or []):
            sdx, sdy = sx2 - sx1, sy2 - sy1
            sl = math.hypot(sdx, sdy)
            if sl < 1:
                continue
            if abs((sdx * ux + sdy * uy) / sl) < 0.97:        # not parallel
                continue
            mx, my = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            if abs((mx - x1) * px + (my - y1) * py) > 6.0:     # not collinear
                continue
            t1 = (sx1 - x1) * ux + (sy1 - y1) * uy
            t2 = (sx2 - x1) * ux + (sy2 - y1) * uy
            if max(t1, t2) < -8 or min(t1, t2) > L + 8:        # doesn't overlap body
                continue
            d_ts += [t1, t2]
        if d_ts:
            new0 = max(new0, min(d_ts) - PAD)     # don't start before drawn steel + PAD
            new1 = min(new1, max(d_ts) + PAD)     # don't end past drawn steel + PAD

        if new1 - new0 < MIN_SPAN or (abs(new0) < 1e-6 and abs(new1 - L) < 1e-6):
            continue

        nx1, ny1 = x1 + ux * new0, y1 + uy * new0
        nx2, ny2 = x1 + ux * new1, y1 + uy * new1
        m["bx1"] = round(nx1 / page_w, 4); m["by1"] = round(ny1 / page_h, 4)
        m["bx2"] = round(nx2 / page_w, 4); m["by2"] = round(ny2 / page_h, 4)
        m["x"]   = round((nx1 + nx2) / 2 / page_w, 4)
        m["y"]   = round((ny1 + ny2) / 2 / page_h, 4)
        m["length_ft"] = round(math.hypot(nx2 - nx1, ny2 - ny1) / ppf, 1)
    return members


def collect_line_widths(page, plan_bounds):
    """Collect drawn SOLID line segments with their stroke WIDTH.

    Returns [(x1, y1, x2, y2, length, width), ...] inside (or near) the plan.

    DASHED lines (curtain-wall boundaries, hidden conditions, reference lines,
    stair-symbol diagonals drawn dashed) are excluded.  Structural members are
    always drawn as SOLID lines; dashed lines are annotations or boundaries.
    """
    out = []
    bx0, by0, bx1, by1 = plan_bounds
    try:
        for d in page.get_drawings():
            # Skip dashed / dotted paths.  PyMuPDF reports solid lines as
            # dashes="" or "[] 0"; anything else is a dash pattern.
            dashes = str(d.get("dashes") or "").strip()
            if dashes and dashes not in ("[] 0", "[] 0.0", "[]"):
                continue                            # dashed → skip
            w = d.get("width")
            wv = float(w) if w is not None else 0.5
            for it in d.get("items", []):
                if it[0] != "l":
                    continue
                try:
                    p1, p2 = it[1], it[2]
                    ln = math.hypot(p2.x - p1.x, p2.y - p1.y)
                    if ln < 20:
                        continue
                    mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
                    if not (bx0 - 300 <= mx <= bx1 + 300 and by0 - 300 <= my <= by1 + 300):
                        continue
                    out.append((p1.x, p1.y, p2.x, p2.y, ln, wv))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def center_beam_overlays(members, lines_w, page_w, page_h):
    """RENDER-ONLY centring (runs AFTER all dedup, so it can never drop a beam).

    The matcher can land on a thin line near the label that sits slightly above
    the actual beam.  Structural beam centrelines are drawn THICK; grid /
    dimension / leader lines are thin.  Here we snap each beam's overlay onto the
    THICKEST parallel drawn line that overlaps its span within the section-depth
    band — i.e. onto the dark beam line itself.  Only the finalised overlay
    (and its chip) is moved; no member is added, removed, or re-deduped.
    """
    if not lines_w:
        return members
    for m in members:
        if m.get("type") != "beam" or m.get("bx1") is None:
            continue
        # Unlabeled candidates are placed at the geometric midpoint between
        # the two detected flange lines — that IS the beam centreline.
        # Snapping to the nearest thick line would move the overlay to one
        # flange (half a section depth off).  The computed midpoint is correct;
        # leave it alone.
        if m.get("unlabeled"):
            continue
        x1 = m["bx1"] * page_w; y1 = m["by1"] * page_h
        x2 = m["bx2"] * page_w; y2 = m["by2"] * page_h
        bdx, bdy = x2 - x1, y2 - y1
        bln = math.hypot(bdx, bdy)
        if bln < 1:
            continue
        ux, uy = bdx / bln, bdy / bln
        px, py = -uy, ux                      # perpendicular
        best_w, best_perp = 0.0, None
        for (qx1, qy1, qx2, qy2, qln, qw) in lines_w:
            if qln < 0.5 * bln:               # must span most of the beam
                continue
            qdx, qdy = qx2 - qx1, qy2 - qy1
            if abs((qdx * ux + qdy * uy) / qln) < 0.97:
                continue                       # not parallel to the beam
            perp = (qx1 - x1) * px + (qy1 - y1) * py
            if abs(perp) > 18.0:               # outside section-depth band
                continue
            qt = ((qx1 + qx2) / 2 - x1) * ux + ((qy1 + qy2) / 2 - y1) * uy
            if not (0.1 * bln <= qt <= 0.9 * bln):
                continue                       # must overlap the span body
            if qw > best_w:
                best_w, best_perp = qw, perp
        # Only snap when a genuinely THICK structural line was found (beam lines
        # are ≥~1 pt; thin grid/dim lines are ~0.25 pt).  On drawings that draw
        # beams thin this never fires — so it can't shift a correct overlay.
        # Unlabeled candidates are themselves the thick line; widen search band
        # so a candidate whose midpoint was detected slightly off still snaps.
        is_unlab = m.get("unlabeled", False)
        perp_band = 28.0 if is_unlab else 18.0
        # Re-scan with the correct band (already scanned above, redo if unlabeled)
        if is_unlab:
            best_w, best_perp = 0.0, None
            for (qx1, qy1, qx2, qy2, qln, qw) in lines_w:
                if qln < 0.4 * bln:
                    continue
                qdx, qdy = qx2 - qx1, qy2 - qy1
                if abs((qdx * ux + qdy * uy) / (qln or 1)) < 0.97:
                    continue
                perp = (qx1 - x1) * px + (qy1 - y1) * py
                if abs(perp) > perp_band:
                    continue
                qt = ((qx1 + qx2) / 2 - x1) * ux + ((qy1 + qy2) / 2 - y1) * uy
                if not (0.05 * bln <= qt <= 0.95 * bln):
                    continue
                if qw > best_w:
                    best_w, best_perp = qw, perp
        min_w = 0.7 if is_unlab else 1.0
        if best_perp is not None and best_w >= min_w and abs(best_perp) > 0.5:
            dx, dy = px * best_perp, py * best_perp
            m["bx1"] = round((x1 + dx) / page_w, 4)
            m["by1"] = round((y1 + dy) / page_h, 4)
            m["bx2"] = round((x2 + dx) / page_w, 4)
            m["by2"] = round((y2 + dy) / page_h, 4)
            if m.get("x") is not None:
                m["x"] = round(m["x"] + dx / page_w, 4)
            if m.get("y") is not None:
                m["y"] = round(m["y"] + dy / page_h, 4)
            # keep label chip (lx/ly) in sync with the centered midpoint
            if m.get("lx") is not None:
                m["lx"] = round(m["lx"] + dx / page_w, 4)
            if m.get("ly") is not None:
                m["ly"] = round(m["ly"] + dy / page_h, 4)
    return members


# ── Request models ────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    filename:         str
    page_index:       int   = 0
    scale_ratio:      float = None
    ocr_dpi:          int   = 400
    detect_unlabeled: bool  = False   # off by default — enable to show (beam?) candidates

class SaveProjectRequest(BaseModel):
    name:        str
    filename:    str
    scale:       str   = None
    scale_ratio: int   = None
    members:     list  = []
    page_count:  int   = 1


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("===== STEELGENIE BACKEND STARTED =====")

@app.get("/health")
def health():
    return {"status": "ok", "supabase": supabase_client is not None}


def _render_page_pil(file_path: str, page_index: int, max_width: int = 2800) -> PILImage.Image:
    """Render a single PDF page (or image file) to a PIL image capped at max_width."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pil = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        pil = PILImage.open(file_path).convert("RGB")
    if pil.width > max_width:
        ratio = max_width / pil.width
        pil = pil.resize((max_width, int(pil.height * ratio)), PILImage.LANCZOS)
    return pil


def _pil_to_b64(pil: PILImage.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content   = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    MAX_PREVIEW_W  = 2800   # full-size preview width cap
    MAX_THUMB_W    = 220    # sidebar thumbnail width cap

    def _render_page_to_b64(pil_img: "PILImage.Image", max_w: int, quality: int, fmt: str = "JPEG") -> str:
        if pil_img.width > max_w:
            ratio = max_w / pil_img.width
            pil_img = pil_img.resize((max_w, int(pil_img.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        if fmt == "PNG":
            pil_img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        pil_img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".pdf":
        doc        = fitz.open(file_path)
        page_count = len(doc)

        # Full-size preview for page 0 (viewed in the drawing window)
        pix0 = doc[0].get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        pil0 = PILImage.frombytes("RGB", [pix0.width, pix0.height], pix0.samples)

        # Small thumbnails for every page (sidebar strip)
        page_thumbnails = []
        for i in range(page_count):
            pix_t = doc[i].get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
            pil_t = PILImage.frombytes("RGB", [pix_t.width, pix_t.height], pix_t.samples)
            page_thumbnails.append(_render_page_to_b64(pil_t, MAX_THUMB_W, 70))

        doc.close()
        main_b64 = _render_page_to_b64(pil0, MAX_PREVIEW_W, 85, fmt="PNG")
    else:
        page_count = 1
        pil = PILImage.open(file_path).convert("RGB")
        main_b64 = _render_page_to_b64(pil, MAX_PREVIEW_W, 85, fmt="PNG")
        page_thumbnails = [_render_page_to_b64(
            pil.resize((MAX_THUMB_W, int(pil.height * MAX_THUMB_W / pil.width)), PILImage.LANCZOS), MAX_THUMB_W, 70)]

    return {
        "image":           main_b64,
        "page_count":      page_count,
        "filename":        file.filename,
        "page_thumbnails": page_thumbnails,
    }


@app.get("/page-image/{filename}/{page_index}")
async def get_page_image(filename: str, page_index: int):
    """Return full-size preview for a specific page of an already-uploaded PDF."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            raise HTTPException(404, f"Page {page_index} not found (total {len(doc)})")
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        pil = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        if page_index != 0:
            raise HTTPException(404, "Image files have only one page")
        pil = PILImage.open(file_path).convert("RGB")

    MAX_PREVIEW_W = 2800
    if pil.width > MAX_PREVIEW_W:
        r = MAX_PREVIEW_W / pil.width
        pil = pil.resize((MAX_PREVIEW_W, int(pil.height * r)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": f"data:image/png;base64,{b64}", "page_index": page_index}


@app.post("/analyse")
async def analyse_pdf(req: AnalysisRequest):
    start = time.time()
    try:
        tmp_path = os.path.join(UPLOAD_DIR, req.filename)
        ext      = os.path.splitext(req.filename)[1].lower()

        doc  = fitz.open(tmp_path)
        page = doc[req.page_index]

        # ── Page dimensions ───────────────────────────────────────────────────
        if ext in _IMAGE_EXTS:
            _pil_tmp = PILImage.open(tmp_path)
            page_w   = float(_pil_tmp.width)
            page_h   = float(_pil_tmp.height)
            _pil_tmp.close()
            print(f"[ANALYSE] Image file — using PIL dims {page_w:.0f}x{page_h:.0f}")
        else:
            # ── Rotation handling ─────────────────────────────────────────────
            # PyMuPDF returns text/drawing coordinates in the UNROTATED page
            # space, but page.rect reports the ROTATED (displayed) dimensions.
            # For a 90°/270° rotated page these disagree (width↔height swapped),
            # so every member normalized by page.rect lands in the wrong place
            # (the bottom of the plan falls off the assumed height).
            #
            # Fix: do ALL internal processing in the unrotated space (mediabox
            # dims), which matches the coordinate space the text/lines are in.
            # The final member coordinates are transformed back into the rotated
            # DISPLAY space at the end (see "ROTATION OUTPUT TRANSFORM" below),
            # because the image the frontend shows IS rotated.
            #
            # Rotation 0 → mediabox == rect → behaviour identical to before, so
            # non-rotated drawings are completely unaffected.
            if page.rotation in (90, 270):
                page_w, page_h = page.mediabox.width, page.mediabox.height
            else:
                page_w, page_h = page.rect.width, page.rect.height

        # ── 0. Text extraction (vector embedded OR OCR for raster images) ─────
        text_dict, is_raster = _get_text_dict(page, page_w, page_h)
        print(f"[ANALYSE] is_raster={is_raster}  text_blocks={len(text_dict.get('blocks', []))}")

        # 1. Plan boundary — excludes schedules, notes, title block
        plan_bounds = find_plan_boundary(page, page_w, page_h, text_dict=text_dict)

        # 2. Detect column symbols (I/H shapes in vector paths) — skip for raster
        column_symbols = [] if is_raster else detect_column_symbols(page)

        # 3. Grid lines — SECONDARY signal
        v_grid, h_grid = extract_grid_lines(page, page_w, page_h, plan_bounds,
                                            text_dict=text_dict)

        # 3b. For raster images, replace text-derived grid with morphological
        #     line detection: finds the actual drawn structural column lines
        #     (long vertical/horizontal lines spanning ≥40% of the plan).
        if is_raster and ext in _IMAGE_EXTS:
            rv, rh = detect_raster_grid_lines(tmp_path, plan_bounds)
            if rv:
                v_grid = rv
            if rh:
                h_grid = rh

        # 3c. If v_grid or h_grid is empty, derive from column symbol positions.
        #     Vertical framing elevations often have no text grid labels, so
        #     extract_grid_lines returns empty lists.  Column symbol X/Y positions
        #     are reliable substitutes: cluster within 30 pt merge tolerance.
        def _cluster_positions(vals, tol=30.0):
            out = []
            for v in sorted(vals):
                if not out or v - out[-1] > tol:
                    out.append(v)
            return out

        if column_symbols:
            if not v_grid:
                _sym_xs = [s["cx"] for s in column_symbols]
                v_grid = _cluster_positions(_sym_xs, tol=30.0)
                if v_grid:
                    print(f"[ANALYSE] v_grid derived from {len(column_symbols)} symbols: "
                          f"{[round(x) for x in v_grid]}")
            if not h_grid:
                _sym_ys = [s["cy"] for s in column_symbols]
                h_grid = _cluster_positions(_sym_ys, tol=30.0)
                if h_grid:
                    print(f"[ANALYSE] h_grid derived from {len(column_symbols)} symbols: "
                          f"{[round(y) for y in h_grid]}")

        # 4. Extract profile labels within the plan
        profiles = extract_profiles(page, page_w, page_h, plan_bounds,
                                    text_dict=text_dict)
        print(f"[ANALYSE] Profiles in plan: {len(profiles)}")

        # ── DIAGNOSTIC DUMP ──────────────────────────────────────────────────
        syms_in_plan = [s for s in column_symbols
                        if plan_bounds[0] <= s["cx"] <= plan_bounds[2]
                        and plan_bounds[1] <= s["cy"] <= plan_bounds[3]]
        print(f"\n{'='*60}")
        print(f"[DIAG] Page size      : {page_w:.0f} x {page_h:.0f} pt")
        # Rotation diagnostics: a /Rotate flag makes page.rect dims disagree with
        # the text/drawing coordinate space, mis-normalizing every member.
        try:
            _mb = page.mediabox
            print(f"[DIAG] Page rotation  : {page.rotation}°")
            print(f"[DIAG] MediaBox       : {_mb.width:.0f} x {_mb.height:.0f} pt")
            _all_cy = [p['cy'] for p in profiles] + [s['cy'] for s in column_symbols]
            _all_cx = [p['cx'] for p in profiles] + [s['cx'] for s in column_symbols]
            if _all_cx and _all_cy:
                print(f"[DIAG] Content extent : x=[{min(_all_cx):.0f},{max(_all_cx):.0f}] "
                      f"y=[{min(_all_cy):.0f},{max(_all_cy):.0f}]  "
                      f"(vs page {page_w:.0f}x{page_h:.0f})")
        except Exception as _e:
            print(f"[DIAG] rotation probe failed: {_e}")
        print(f"[DIAG] Raster mode    : {is_raster}")
        print(f"[DIAG] Plan boundary  : x=[{plan_bounds[0]:.0f},{plan_bounds[2]:.0f}] "
              f"y=[{plan_bounds[1]:.0f},{plan_bounds[3]:.0f}]")
        print(f"[DIAG] V grid lines   : {len(v_grid)}  -> {[round(x) for x in v_grid]}")
        print(f"[DIAG] H grid lines   : {len(h_grid)}  -> {[round(y) for y in h_grid]}")
        print(f"[DIAG] Symbols total  : {len(column_symbols)}  "
              f"| inside plan: {len(syms_in_plan)}")
        for s in column_symbols:
            tag = "IN " if (plan_bounds[0] <= s["cx"] <= plan_bounds[2]
                            and plan_bounds[1] <= s["cy"] <= plan_bounds[3]) else "OUT"
            print(f"  [{tag}] cx={s['cx']:.0f}  cy={s['cy']:.0f}")
        print(f"[DIAG] Profiles found : {len(profiles)}")
        for i, p in enumerate(profiles):
            print(f"  [{i:02d}] {p['profile']:14s}  cx={p['cx']:.0f}  cy={p['cy']:.0f}")
        print(f"{'='*60}\n")
        # ─────────────────────────────────────────────────────────────────────

        # 5. Scale → pts per foot (from user-selected drawing scale)
        pts_per_foot = scale_to_pts_per_foot(req.scale_ratio) if req.scale_ratio else 0.0
        print(f"[ANALYSE] scale_ratio={req.scale_ratio}  pts_per_foot={pts_per_foot:.2f}")

        # 6. PRIMARY: match beam centerlines to profile labels
        #    • Vector PDF: read page.get_drawings() — exact mathematical geometry.
        #    • Raster image: render to 300-DPI greyscale and run HoughLinesP.
        #      Before this fix the raster path skipped this step entirely
        #      (beam_line_map = {}), forcing 100 % of beams through the
        #      grid-based fallback which only achieves bay-level precision.
        if is_raster and _RASTER_HOUGH_AVAILABLE and profiles:
            _DPI_HOUGH  = 300
            _PX_PER_PT  = _DPI_HOUGH / 72.0
            _pix        = page.get_pixmap(dpi=_DPI_HOUGH, colorspace=fitz.csGRAY)
            _gray_arr   = np.frombuffer(_pix.samples, dtype=np.uint8).reshape(
                              _pix.height, _pix.width)
            beam_line_map = _detect_beam_lines_raster(
                _gray_arr, profiles, plan_bounds, _PX_PER_PT)
            print(f"[ANALYSE] Raster Hough beam_line_map: {len(beam_line_map)} hits")
            del _pix, _gray_arr   # free memory
        else:
            if is_raster:
                beam_line_map   = {}
                _all_struct_lns = []
            else:
                beam_line_map, _all_struct_lns = detect_beam_lines(
                    page, profiles, plan_bounds,
                    pts_per_foot=pts_per_foot,
                    column_symbols=column_symbols,
                    v_grid=v_grid,
                    h_grid=h_grid)

        # ── Unlabeled beam detection ──────────────────────────────────────────
        # Only runs when the drawing has ZERO profile labels (e.g. a framing plan
        # where the engineer drew all beam lines but added no text tags).
        # When labeled profiles exist we skip this entirely — labeled detection is
        # already complete and we don't want to double-count.
        #
        # Safety filters applied to every candidate line:
        #   F1  Orthogonal only (H or V) — unlabeled diagonal lines are
        #       indistinguishable from braces, so we skip them.
        #   F2  Length in structural range: MIN_LEN–MAX_LEN (already applied in
        #       detect_beam_lines, so all_struct_lns satisfies this).
        #   F3  Does NOT span >85 % of the plan width (H) or plan height (V) —
        #       full-width lines are grid / dimension lines, not beams.
        #   F4  Deduplicate: skip if another synthetic is already within 40 pt.
        #   F5  Cap at 200 synthetics to prevent memory issues on dense drawings.
        if not is_raster and len(profiles) == 0 and _all_struct_lns:
            _pb = plan_bounds
            _plan_w = _pb[2] - _pb[0]
            _plan_h = _pb[3] - _pb[1]
            _MAX_H_FRAC = 0.85   # F3: H-line must be shorter than 85 % of plan width
            _MAX_V_FRAC = 0.85   # F3: V-line must be shorter than 85 % of plan height
            _DEDUP_R    = 40     # F4: midpoint deduplication radius
            _SYN_CAP    = 200    # F5: hard cap

            _syn_profiles  = []
            _syn_line_map  = {}
            _seen_mids: list[tuple] = []

            for (lx1, ly1, lx2, ly2, ln) in _all_struct_lns:
                if len(_syn_profiles) >= _SYN_CAP:
                    break
                adx = abs(lx2 - lx1)
                ady = abs(ly2 - ly1)

                # F1 — orthogonal only
                if adx > ady * 2:
                    bdir = "H"
                elif ady > adx * 2:
                    bdir = "V"
                else:
                    continue   # diagonal — skip

                # F3 — reject full-plan-width / full-plan-height lines
                if bdir == "H" and adx > _plan_w * _MAX_H_FRAC:
                    continue
                if bdir == "V" and ady > _plan_h * _MAX_V_FRAC:
                    continue

                mx = (lx1 + lx2) / 2
                my = (ly1 + ly2) / 2

                # F4 — deduplicate
                if any(math.hypot(smx - mx, smy - my) < _DEDUP_R
                       for smx, smy in _seen_mids):
                    continue

                syn_idx = len(_syn_profiles)
                _syn_profiles.append({
                    "profile":  "?",
                    "cx": mx, "cy": my,
                    "dir_hint": bdir,
                    "bbox_w": 20.0, "bbox_h": 8.0,
                })
                _syn_line_map[syn_idx] = {
                    "x1": lx1, "y1": ly1,
                    "x2": lx2, "y2": ly2,
                    "dir": bdir, "length_pt": ln,
                }
                _seen_mids.append((mx, my))

            if _syn_profiles:
                print(f"[ANALYSE] {len(_syn_profiles)} unlabeled beam lines detected "
                      f"(zero labeled profiles on this page)")
                profiles    = _syn_profiles
                beam_line_map = _syn_line_map

        # 7. FALLBACK direction detection
        #    • Vector: use adjacent drawn lines via detect_beam_directions().
        #    • Raster:  Hough hits already carry a 'dir' key; OCR rotation hints
        #      (dir_hint) fill in for the remaining unmatched profiles.
        if is_raster:
            # Build beam_dirs from Hough results first, then fill gaps from dir_hint
            beam_dirs: dict[int, str] = {}
            for p_idx, hit in beam_line_map.items():
                beam_dirs[p_idx] = hit["dir"]
            # Any profile not matched by Hough falls back to OCR rotation hint
            for p_idx, p in enumerate(profiles):
                if p_idx not in beam_dirs:
                    beam_dirs[p_idx] = p.get("dir_hint", "H")
            h_hints = sum(1 for d in beam_dirs.values() if d == "H")
            v_hints = sum(1 for d in beam_dirs.values() if d == "V")
            print(f"[ANALYSE] Raster beam_dirs (Hough+hint): H={h_hints}  V={v_hints}")
        else:
            beam_dirs = detect_beam_directions(page, profiles, plan_bounds)

        # 8. Classify and build
        members = build_members(profiles, page_w, page_h,
                                column_symbols=column_symbols,
                                v_grid=v_grid, h_grid=h_grid,
                                pts_per_foot=pts_per_foot,
                                beam_dirs=beam_dirs,
                                beam_line_map=beam_line_map,
                                plan_bounds=plan_bounds,
                                is_vector=not is_raster)

        # Trim beams that overshoot the perpendicular girder/beam they frame
        # into — they must STOP at that connection, not pass through it.
        # Column-aware against a CLEAN column set, so an endpoint AT a real
        # column reaches column-centre while girder-overshoots are trimmed.
        # Only shortens overshoots; never lengthens (so correct spans are safe).
        if not is_raster:
            _ccx, _ccy = clean_column_lines(v_grid, h_grid, column_symbols)
            members = trim_beam_overshoot(members, page_w, page_h, pts_per_foot,
                                          col_x=_ccx, col_y=_ccy)

        # Count columns: emit a column member for each column symbol at a grid
        # intersection (framing-plan columns have no size label of their own).
        members = emit_symbol_columns(members, column_symbols, v_grid, h_grid,
                                      page_w, page_h)

        # Unlabeled beams: geometry-detected candidates with no section callout.
        # Gated by detect_unlabeled flag (default OFF) so the UI stays clean
        # unless the user explicitly requests candidate overlay.
        # UNIVERSAL path: flags every uncovered structural beam line (any label
        # count) — not just the conservative double-line/single-line subset.
        _lines_w = collect_line_widths(page, plan_bounds) if not is_raster else []
        if not is_raster and req.detect_unlabeled:
            members = add_unlabeled_lines_universal(
                members, _all_struct_lns,
                plan_bounds, page_w, page_h, pts_per_foot,
                v_grid=v_grid, h_grid=h_grid,
                column_symbols=column_symbols)

        # Universal endpoint snap: pull EVERY beam end (labeled + unlabeled) to the
        # nearest perpendicular support it reaches — column line OR crossing girder
        # — and stop there.  Trims overshoot and closes short gaps along the axis.
        # Runs after unlabeled detection so candidates terminate like real girders.
        if not is_raster:
            _ccx2, _ccy2 = clean_column_lines(v_grid, h_grid, column_symbols)
            members = snap_beam_ends_to_supports(
                members, _all_struct_lns, _ccx2, _ccy2,
                page_w, page_h, pts_per_foot)

        # Render-only: snap EVERY overlay line (labeled + unlabeled candidates)
        # onto the THICK (dark) beam centreline.  Runs LAST so candidates are
        # centred too — they lie ON the beam, not near it.
        if not is_raster:
            members = center_beam_overlays(members, _lines_w, page_w, page_h)

        # ── ROTATION OUTPUT TRANSFORM ─────────────────────────────────────────
        # All member coordinates above are fractions of the UNROTATED page
        # space (page_w × page_h = mediabox for rotated pages).  The image the
        # frontend displays is RENDERED ROTATED (get_pixmap applies the page
        # rotation), so we map each fractional coordinate from unrotated space
        # into the rotated DISPLAY space using fitz's own rotation matrix —
        # this is guaranteed to match how the image was rendered (no manual
        # CW/CCW derivation).  Skipped entirely when rotation == 0.
        if not is_raster and page.rotation != 0:
            _rmat = page.rotation_matrix          # unrotated → rotated coords
            _rw   = page.rect.width               # rotated display dims
            _rh   = page.rect.height
            _puw, _puh = page_w, page_h           # unrotated processing dims

            def _rot_frac(fx, fy):
                if fx is None or fy is None:
                    return fx, fy
                _pt = fitz.Point(fx * _puw, fy * _puh) * _rmat
                return round(_pt.x / _rw, 4), round(_pt.y / _rh, 4)

            for _m in members:
                _m["x"],   _m["y"]   = _rot_frac(_m.get("x"),   _m.get("y"))
                _m["lx"],  _m["ly"]  = _rot_frac(_m.get("lx"),  _m.get("ly"))
                _m["sx"],  _m["sy"]  = _rot_frac(_m.get("sx"),  _m.get("sy"))
                _m["bx1"], _m["by1"] = _rot_frac(_m.get("bx1"), _m.get("by1"))
                _m["bx2"], _m["by2"] = _rot_frac(_m.get("bx2"), _m.get("by2"))
            print(f"[ANALYSE] Applied {page.rotation}° rotation transform "
                  f"to {len(members)} members (unrotated {_puw:.0f}x{_puh:.0f} "
                  f"→ display {_rw:.0f}x{_rh:.0f})")

        summary = build_summary(members)

        counts  = {t: summary[t] for t in ["column", "beam", "vertical_brace"]}
        elapsed = round(time.time() - start, 2)
        method  = ("pymupdf+ocr+hough+grid" if (is_raster and _RASTER_HOUGH_AVAILABLE)
                   else "pymupdf+ocr+grid"   if is_raster
                   else "pymupdf+symbols+grid")
        print(f"[ANALYSE] {len(members)} members in {elapsed}s — {counts}")

        doc.close()
        return {
            "members":         members,
            "summary":         summary,
            "method":          method,
            "elapsed":         elapsed,
            "elapsed_seconds": elapsed,
            "count":           len(members),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/projects")
async def get_projects():
    if not supabase_client:
        raise HTTPException(503, "Supabase not configured")
    return supabase_client.table("projects").select("*, files(*)").execute().data


_SAVED_FILE = os.path.join(BASE_DIR, "saved_projects.json")


@app.post("/save-project")
async def save_project(req: SaveProjectRequest):
    import json as _json, uuid as _uuid

    projects: list = []
    if os.path.exists(_SAVED_FILE):
        try:
            with open(_SAVED_FILE, "r", encoding="utf-8") as f:
                projects = _json.load(f)
        except Exception:
            projects = []

    entry = {
        "id":           str(_uuid.uuid4()),
        "name":         req.name,
        "filename":     req.filename,
        "scale":        req.scale,
        "scale_ratio":  req.scale_ratio,
        "members":      req.members,
        "member_count": len(req.members),
        "page_count":   req.page_count,
        "created_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    projects.insert(0, entry)

    with open(_SAVED_FILE, "w", encoding="utf-8") as f:
        _json.dump(projects, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "id": entry["id"]}


@app.get("/saved-projects")
async def get_saved_projects():
    import json as _json

    if not os.path.exists(_SAVED_FILE):
        return []
    try:
        with open(_SAVED_FILE, "r", encoding="utf-8") as f:
            projects = _json.load(f)
        return [
            {
                "id":           p.get("id"),
                "name":         p.get("name"),
                "filename":     p.get("filename", ""),
                "scale":        p.get("scale", ""),
                "member_count": p.get("member_count", 0),
                "page_count":   p.get("page_count", 1),
                "created_at":   p.get("created_at", ""),
            }
            for p in projects
        ]
    except Exception:
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
