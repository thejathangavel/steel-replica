"""
Brace Duplicate Analysis
-------------------------
Investigates duplicate brace members found during integration verification.
Affected pages:
  - Structural snaps.pdf  page 6  (4 duplicates)
  - Structural snaps.pdf  page 7  (7 duplicates)

For each duplicate pair:
  1. Prints raw PDF drawing coordinates
  2. Measures endpoint delta (pts and fractional)
  3. Renders visual evidence image (side-by-side duplicate segments)

Outputs:
  brace_dedup_analysis/
    01_snaps_p6_overview.jpg       -- full page overlay showing all braces
    02_snaps_p6_dupNN_crop.jpg     -- zoomed crop per duplicate pair
    ...
  brace_dedup_report_raw.json      -- machine-readable findings
"""
import os, sys, io, math, json
import fitz
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUT_DIR    = os.path.join(HERE, "brace_dedup_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── pages to investigate (those with duplicates from verify_integration) ────────
TARGETS = [
    ("snaps_p6", "Structural snaps.pdf", 6, 96.0),
    ("snaps_p7", "Structural snaps.pdf", 7, 96.0),
]

DEDUP_THRESHOLD_FRAC = 0.015  # fractional units — pairs closer than this are duplicates


# ── raw extraction WITHOUT deduplication ─────────────────────────────────────────
import re as _re
import math as _math

def extract_diagonals_raw(page, ppf, min_ft=3.0):
    """Same as brace_classifier.extract_diagonals but NO deduplication."""
    min_pt  = min_ft * ppf
    results = []
    for drawing_idx, d in enumerate(page.get_drawings()):
        dashes = str(d.get("dashes") or "").strip()
        da = _re.match(r'\[([^\]]*)\]', dashes)
        if da and da.group(1).strip():
            continue
        sw = float(d.get("width") or 0)
        color = d.get("color") or d.get("stroke_color")
        for item in d.get("items", []):
            if item[0] != "l":
                continue
            try:
                p1, p2 = item[1], item[2]
                dx, dy = p2.x - p1.x, p2.y - p1.y
                ln = _math.hypot(dx, dy)
                if ln < min_pt:
                    continue
                ang = abs(_math.degrees(_math.atan2(dy, dx))) % 180
                ah  = min(ang, 180 - ang)
                if not (20.0 <= ah <= 70.0):
                    continue
                results.append({
                    "x1": p1.x, "y1": p1.y,
                    "x2": p2.x, "y2": p2.y,
                    "length_ft": round(ln / ppf, 2),
                    "angle_from_h": round(ah, 1),
                    "stroke_w": sw,
                    "drawing_idx": drawing_idx,
                    "color": str(color),
                })
            except Exception:
                continue
    return results


def ppf(scale): return 864.0 / scale if scale > 0 else 9.0


def find_duplicate_pairs(candidates, page_w, page_h, threshold=DEDUP_THRESHOLD_FRAC):
    """
    Find pairs of candidates whose fractional endpoint distances are < threshold.
    Returns list of (i, j, delta_fwd, delta_rev) tuples.
    """
    pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            # Forward distance
            dfwd = (math.hypot((a["x1"] - b["x1"]) / page_w, (a["y1"] - b["y1"]) / page_h) +
                    math.hypot((a["x2"] - b["x2"]) / page_w, (a["y2"] - b["y2"]) / page_h))
            # Reversed distance (b drawn in opposite direction)
            drev = (math.hypot((a["x1"] - b["x2"]) / page_w, (a["y1"] - b["y2"]) / page_h) +
                    math.hypot((a["x2"] - b["x1"]) / page_w, (a["y2"] - b["y1"]) / page_h))
            d = min(dfwd, drev)
            if d < threshold:
                pairs.append({
                    "i": i, "j": j,
                    "distance_frac": round(d, 5),
                    "direction": "forward" if dfwd <= drev else "reversed",
                    "delta_x1_pt": round(abs(a["x1"] - b["x1"]) if dfwd <= drev else abs(a["x1"] - b["x2"]), 3),
                    "delta_y1_pt": round(abs(a["y1"] - b["y1"]) if dfwd <= drev else abs(a["y1"] - b["y2"]), 3),
                    "delta_x2_pt": round(abs(a["x2"] - b["x2"]) if dfwd <= drev else abs(a["x2"] - b["x1"]), 3),
                    "delta_y2_pt": round(abs(a["y2"] - b["y2"]) if dfwd <= drev else abs(a["y2"] - b["y1"]), 3),
                    "same_drawing": candidates[i]["drawing_idx"] == candidates[j]["drawing_idx"],
                    "same_stroke_w": round(abs(candidates[i]["stroke_w"] - candidates[j]["stroke_w"]), 4),
                    "length_diff_ft": round(abs(candidates[i]["length_ft"] - candidates[j]["length_ft"]), 3),
                })
    return pairs


def get_font(size=11):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(name, size)
        except: pass
    return ImageFont.load_default()


def render_overview(page, candidates, pairs, dpi=80, title=""):
    """Full-page overlay. Unique braces = green, duplicates = red/magenta pairs."""
    pix   = page.get_pixmap(dpi=dpi)
    img   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw  = ImageDraw.Draw(img)
    s     = dpi / 72.0
    font  = get_font(9)

    dup_indices = set()
    for p in pairs:
        dup_indices.add(p["i"])
        dup_indices.add(p["j"])

    # Draw non-duplicates green
    for idx, c in enumerate(candidates):
        if idx not in dup_indices:
            x1, y1 = c["x1"]*s, c["y1"]*s
            x2, y2 = c["x2"]*s, c["y2"]*s
            draw.line([(x1,y1),(x2,y2)], fill=(0,220,80), width=2)

    # Draw duplicate pairs in red (first) and magenta (second)
    colors = [(255,60,60), (255,0,220), (255,180,0), (0,200,255),
              (180,0,255), (255,120,0), (0,255,180)]
    pair_color = {}
    for pidx, p in enumerate(pairs):
        col = colors[pidx % len(colors)]
        pair_color[p["i"]] = col
        pair_color[p["j"]] = col

    for idx in dup_indices:
        c  = candidates[idx]
        x1, y1 = c["x1"]*s, c["y1"]*s
        x2, y2 = c["x2"]*s, c["y2"]*s
        col = pair_color.get(idx, (255,0,0))
        draw.line([(x1,y1),(x2,y2)], fill=col, width=4)
        r = 4
        draw.ellipse([(x1-r,y1-r),(x1+r,y1+r)], outline=col, width=2)
        draw.ellipse([(x2-r,y2-r),(x2+r,y2+r)], outline=col, width=2)
        draw.text((x1+4, y1-10), f"#{idx}", fill=col, font=font)

    # Title bar
    draw.rectangle([(0,0),(img.width,16)], fill=(10,10,10))
    draw.text((4,2), f"{title}  |  {len(candidates)} candidates  |  "
              f"{len(pairs)} dup pairs  |  GREEN=unique  COLORED=dup", fill=(200,200,200), font=font)
    return img


def render_dup_crop(page, a, b, pair_info, dpi=150, page_w=1, page_h=1):
    """
    Zoomed crop showing both members of a duplicate pair.
    Member A = red, Member B = magenta.
    """
    # Crop rect centred between the two midpoints
    points = [a["x1"], a["y1"], a["x2"], a["y2"], b["x1"], b["y1"], b["x2"], b["y2"]]
    xs = points[0::2]; ys = points[1::2]
    cx = sum(xs)/4; cy = sum(ys)/4
    half = max(200, (max(xs)-min(xs))*0.8 + 120, (max(ys)-min(ys))*0.8 + 120)

    pr   = page.rect
    clip = fitz.Rect(
        max(cx - half, pr.x0), max(cy - half, pr.y0),
        min(cx + half, pr.x1), min(cy + half, pr.y1)
    )
    if clip.width <= 1 or clip.height <= 1:
        clip = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y1)

    pix  = page.get_pixmap(dpi=dpi, clip=clip)
    if pix.width == 0 or pix.height == 0:
        raise ValueError("Empty pixmap")
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    s    = dpi / 72.0
    font = get_font(11)

    def to_pix(xp, yp):
        return (xp - clip.x0)*s, (yp - clip.y0)*s

    # Draw member A (red, wider)
    ax1,ay1 = to_pix(a["x1"], a["y1"])
    ax2,ay2 = to_pix(a["x2"], a["y2"])
    draw.line([(ax1,ay1),(ax2,ay2)], fill=(255,60,60), width=5)
    r=7
    draw.ellipse([(ax1-r,ay1-r),(ax1+r,ay1+r)], outline=(255,60,60), width=3)
    draw.ellipse([(ax2-r,ay2-r),(ax2+r,ay2+r)], outline=(255,60,60), width=3)
    draw.text((ax1+r+2, ay1-8), "A", fill=(255,60,60), font=get_font(13))

    # Draw member B (magenta, slightly offset for visibility)
    bx1,by1 = to_pix(b["x1"], b["y1"])
    bx2,by2 = to_pix(b["x2"], b["y2"])
    draw.line([(bx1+2,by1+2),(bx2+2,by2+2)], fill=(255,0,220), width=3)
    r=5
    draw.ellipse([(bx1-r,by1-r),(bx1+r,by1+r)], outline=(255,0,220), width=2)
    draw.ellipse([(bx2-r,by2-r),(bx2+r,by2+r)], outline=(255,0,220), width=2)
    draw.text((bx1+r+2, by1+4), "B", fill=(255,0,220), font=get_font(13))

    # Info bar
    info = (f"A: ({a['x1']:.2f},{a['y1']:.2f})->({a['x2']:.2f},{a['y2']:.2f})  "
            f"B: ({b['x1']:.2f},{b['y1']:.2f})->({b['x2']:.2f},{b['y2']:.2f})  "
            f"d={pair_info['distance_frac']:.5f}  "
            f"dx1={pair_info['delta_x1_pt']:.2f}pt  dy1={pair_info['delta_y1_pt']:.2f}pt  "
            f"same_drw={pair_info['same_drawing']}")
    draw.rectangle([(0,0),(img.width,18)], fill=(10,10,10))
    draw.text((4,2), info, fill=(200,220,200), font=get_font(10))

    return img


def save_img(img, path):
    jpg = path.replace(".png", ".jpg")
    img.convert("RGB").save(jpg, format="JPEG", quality=92)
    return jpg


# ── Main analysis ─────────────────────────────────────────────────────────────────
def run():
    print("=" * 76)
    print("  BRACE DUPLICATE ANALYSIS")
    print("=" * 76)

    all_findings = []

    for (label, pdf_name, pg_idx, scale) in TARGETS:
        path = os.path.join(UPLOAD_DIR, pdf_name)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {pdf_name} not found")
            continue

        pts_per_foot = ppf(scale)
        doc  = fitz.open(path)
        page = doc[pg_idx]
        pw, ph = page.rect.width, page.rect.height

        print(f"\n{'─'*76}")
        print(f"  [{label}]  {pdf_name}  page={pg_idx}  scale=1:{scale:.0f}  "
              f"page={pw:.0f}x{ph:.0f}pt  ppf={pts_per_foot:.2f}")
        print(f"{'─'*76}")

        # Raw extraction (no dedup)
        raw = extract_diagonals_raw(page, pts_per_foot)
        print(f"  Raw diagonal candidates (no dedup): {len(raw)}")

        # Find duplicates
        pairs = find_duplicate_pairs(raw, pw, ph)
        print(f"  Duplicate pairs found: {len(pairs)}")

        if not pairs:
            print("  No duplicates on this page.")
            doc.close()
            continue

        # Detail each pair
        print()
        print(f"  {'Pair':<6}  {'Indices':<8}  {'Dist(frac)':<12}  "
              f"{'dx1 pt':<8}  {'dy1 pt':<8}  {'dx2 pt':<8}  {'dy2 pt':<8}  "
              f"{'SameDrw':<8}  {'dLen ft':<8}  {'Direction'}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*12}  "
              f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  "
              f"{'─'*8}  {'─'*8}  {'─'*9}")

        for pidx, p in enumerate(pairs):
            a = raw[p["i"]]
            b = raw[p["j"]]
            print(f"  {pidx+1:<6}  ({p['i']},{p['j']})  "
                  f"{p['distance_frac']:<12.5f}  "
                  f"{p['delta_x1_pt']:<8.2f}  {p['delta_y1_pt']:<8.2f}  "
                  f"{p['delta_x2_pt']:<8.2f}  {p['delta_y2_pt']:<8.2f}  "
                  f"{str(p['same_drawing']):<8}  {p['length_diff_ft']:<8.3f}  "
                  f"{p['direction']}")
            print(f"         A: ({a['x1']:.3f},{a['y1']:.3f})->({a['x2']:.3f},{a['y2']:.3f})  "
                  f"len={a['length_ft']}ft  sw={a['stroke_w']}  drw={a['drawing_idx']}")
            print(f"         B: ({b['x1']:.3f},{b['y1']:.3f})->({b['x2']:.3f},{b['y2']:.3f})  "
                  f"len={b['length_ft']}ft  sw={b['stroke_w']}  drw={b['drawing_idx']}")

        # Max delta across all pairs
        max_dx1 = max(p["delta_x1_pt"] for p in pairs)
        max_dy1 = max(p["delta_y1_pt"] for p in pairs)
        max_dx2 = max(p["delta_x2_pt"] for p in pairs)
        max_dy2 = max(p["delta_y2_pt"] for p in pairs)
        print(f"\n  Max endpoint deltas across all pairs:")
        print(f"    dx1={max_dx1:.3f}pt  dy1={max_dy1:.3f}pt  "
              f"dx2={max_dx2:.3f}pt  dy2={max_dy2:.3f}pt")
        print(f"  => Dedup bin size needed: {max(max_dx1,max_dy1,max_dx2,max_dy2):.1f}pt  "
              f"(safety margin: x2 = {max(max_dx1,max_dy1,max_dx2,max_dy2)*2:.1f}pt)")

        # Determine root cause per pair
        same_drawing_count   = sum(1 for p in pairs if p["same_drawing"])
        diff_drawing_count   = len(pairs) - same_drawing_count
        reversed_count       = sum(1 for p in pairs if p["direction"] == "reversed")
        near_zero_delta_count = sum(
            1 for p in pairs
            if max(p["delta_x1_pt"], p["delta_y1_pt"],
                   p["delta_x2_pt"], p["delta_y2_pt"]) < 2.0
        )
        print(f"\n  Root cause indicators:")
        print(f"    Same drawing_idx:      {same_drawing_count}/{len(pairs)}")
        print(f"    Different drawing_idx: {diff_drawing_count}/{len(pairs)}")
        print(f"    Reversed direction:    {reversed_count}/{len(pairs)}")
        print(f"    All deltas < 2pt:      {near_zero_delta_count}/{len(pairs)}")

        # ── Visual evidence ────────────────────────────────────────────────────────
        print(f"\n  Rendering images...")

        # Overview
        ov_img  = render_overview(page, raw, pairs, dpi=80, title=f"{label} p{pg_idx}")
        ov_path = save_img(ov_img, os.path.join(OUT_DIR, f"{label}_overview.png"))
        print(f"    Overview : {ov_path}")

        # Crop per pair
        crop_paths = []
        for pidx, p in enumerate(pairs):
            a = raw[p["i"]]
            b = raw[p["j"]]
            try:
                cr_img  = render_dup_crop(page, a, b, p, dpi=150, page_w=pw, page_h=ph)
                cr_path = save_img(cr_img,
                    os.path.join(OUT_DIR, f"{label}_dup{pidx+1:02d}_({p['i']},{p['j']})_crop.png"))
                crop_paths.append(cr_path)
                print(f"    Dup {pidx+1:02d}   : {cr_path}")
            except Exception as e:
                print(f"    Dup {pidx+1:02d}   : RENDER ERROR — {e}")
                crop_paths.append("RENDER_ERROR")

        doc.close()

        all_findings.append({
            "label":            label,
            "pdf":              pdf_name,
            "page_index":       pg_idx,
            "scale_ratio":      scale,
            "page_w_pt":        pw,
            "page_h_pt":        ph,
            "pts_per_foot":     pts_per_foot,
            "raw_candidates":   len(raw),
            "duplicate_pairs":  len(pairs),
            "max_endpoint_delta_pt": max(max_dx1, max_dy1, max_dx2, max_dy2),
            "same_drawing_pairs": same_drawing_count,
            "diff_drawing_pairs": diff_drawing_count,
            "reversed_pairs":   reversed_count,
            "pairs_detail":     pairs,
            "overview_image":   ov_path,
            "crop_images":      crop_paths,
        })

    # ── Global summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  GLOBAL ROOT CAUSE SUMMARY")
    print(f"{'='*76}")

    total_pairs = sum(f["duplicate_pairs"] for f in all_findings)
    max_delta   = max((f["max_endpoint_delta_pt"] for f in all_findings), default=0)
    same_drw    = sum(f["same_drawing_pairs"] for f in all_findings)
    diff_drw    = sum(f["diff_drawing_pairs"] for f in all_findings)
    rev         = sum(f["reversed_pairs"] for f in all_findings)

    print(f"\n  Total duplicate pairs   : {total_pairs}")
    print(f"  Max endpoint delta (pt) : {max_delta:.3f}")
    print(f"  Same drawing_idx pairs  : {same_drw}")
    print(f"  Diff drawing_idx pairs  : {diff_drw}")
    print(f"  Reversed-direction pairs: {rev}")

    if max_delta > 0:
        bin_needed = max_delta * 2.5
        print(f"\n  DEDUP FIX: round each coordinate to nearest {bin_needed:.1f}pt bin")
        print(f"  (current bin = 1pt — need ~{bin_needed:.0f}pt)")

    # Save JSON
    out_json = os.path.join(HERE, "brace_dedup_report_raw.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_findings, f, indent=2)
    print(f"\n  Raw findings : {out_json}")
    print(f"  Images in    : {OUT_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    run()
