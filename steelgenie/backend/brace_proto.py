"""
Geometry-Based Brace Detection Prototype
-----------------------------------------
Extracts diagonal line candidates from known brace-containing drawings using
SVG geometry only — no labels, no columns, no structural graph.

Goal: determine whether real braces separate naturally from noise by angle +
      length alone.

Output:
  - Console diagnostic table per page (all candidates, ranked by length)
  - PNG overlay image per page (candidates coloured by length tier)

Usage:
  python brace_proto.py
"""
import os, sys, math, json
import fitz
from PIL import Image, ImageDraw, ImageFont

HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUT_DIR    = os.path.join(HERE, "brace_proto_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Test pages — chosen because analysis confirmed diagonal lines + brace labels
TEST_PAGES = [
    # name,                  file,                                          page, scale_ratio
    ("binder_p4_braceframe", "#Structural binder.pdf",                       4,  64.0),
    ("binder_p7_braceframe", "#Structural binder.pdf",                       7,  64.0),
    ("binder_p8_braceframe", "#Structural binder.pdf",                       8,  64.0),
    ("snaps_p1_elevation",   "Structural snaps.pdf",                         1,  96.0),
    ("snaps_p6_elevation",   "Structural snaps.pdf",                         6,  96.0),
    ("snaps_p3_framing",     "Structural snaps.pdf",                         3,  96.0),
    ("combined_p2_notes",    "07_STRUCTURAL_COMBINED.pdf",                   2, 192.0),
    ("combined_p23_plan",    "07_STRUCTURAL_COMBINED.pdf",                  23, 192.0),
]

# ── Length thresholds to test (in feet) ──────────────────────────────────────
# We show candidates at all thresholds so we can see the distribution
RENDER_MIN_FT  = 3.0   # draw everything ≥ 3ft so we can see what's there
REPORT_BUCKETS = [3, 5, 7, 10, 15, 20]   # count how many survive each cutoff

# ── Colour scheme by length (for overlay image) ───────────────────────────────
def candidate_color(length_ft):
    if length_ft >= 20:  return (255,  80,  80)   # red    — long, likely structural
    if length_ft >= 10:  return (255, 165,   0)   # orange — medium, probable structural
    if length_ft >=  7:  return (255, 230,   0)   # yellow — borderline
    return                      (180, 180, 255)   # blue   — short, likely noise


def scale_to_pts_per_foot(scale_ratio):
    """864 / scale_ratio  (e.g. scale=96 → 9.0 pts/ft)."""
    return 864.0 / scale_ratio if scale_ratio > 0 else 9.0


def extract_diagonal_candidates(page, ppf, min_ft=RENDER_MIN_FT):
    """
    Return all solid, non-dashed, diagonal (20-70° from H) line segments
    on the page with length >= min_ft.

    Each candidate dict:
      x1, y1, x2, y2   — page-pt coordinates
      length_pt         — length in PDF points
      length_ft         — length in feet
      angle_from_h      — degrees from horizontal (0=H, 90=V)
      stroke_w          — line width in pts
      midpoint          — (mx, my)
    """
    min_pt = min_ft * ppf
    candidates = []
    seen = set()   # deduplicate identical segments

    try:
        for d in page.get_drawings():
            # Skip dashed lines
            dashes = str(d.get("dashes") or "").strip()
            import re
            da = re.match(r'\[([^\]]*)\]', dashes)
            if da and da.group(1).strip():
                continue

            stroke_w = float(d.get("width") or 0)

            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                try:
                    p1, p2 = item[1], item[2]
                    dx, dy = p2.x - p1.x, p2.y - p1.y
                    ln = math.hypot(dx, dy)
                    if ln < min_pt:
                        continue

                    # Angle from horizontal
                    ang   = abs(math.degrees(math.atan2(dy, dx))) % 180
                    ang_h = min(ang, 180 - ang)   # 0=H, 90=V
                    if not (20.0 <= ang_h <= 70.0):
                        continue

                    # Deduplicate (same endpoints seen twice from grouped paths)
                    key = (round(p1.x), round(p1.y), round(p2.x), round(p2.y))
                    rkey = (round(p2.x), round(p2.y), round(p1.x), round(p1.y))
                    if key in seen or rkey in seen:
                        continue
                    seen.add(key)

                    candidates.append({
                        "x1": p1.x, "y1": p1.y,
                        "x2": p2.x, "y2": p2.y,
                        "length_pt":    ln,
                        "length_ft":    round(ln / ppf, 2),
                        "angle_from_h": round(ang_h, 1),
                        "stroke_w":     stroke_w,
                        "midpoint":     ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2),
                    })
                except Exception:
                    continue
    except Exception:
        pass

    candidates.sort(key=lambda c: c["length_ft"], reverse=True)
    return candidates


def render_overlay(page, candidates, ppf, label, dpi=120):
    """
    Render the PDF page and draw diagonal candidates on top.
    Returns a PIL Image.
    """
    pix    = page.get_pixmap(dpi=dpi)
    img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw   = ImageDraw.Draw(img)
    scale  = dpi / 72.0   # pts → pixels

    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    # Draw each candidate
    for c in candidates:
        x1 = c["x1"] * scale;  y1 = c["y1"] * scale
        x2 = c["x2"] * scale;  y2 = c["y2"] * scale
        col   = candidate_color(c["length_ft"])
        thick = 3 if c["length_ft"] >= 10 else 2

        draw.line([(x1, y1), (x2, y2)], fill=col, width=thick)

        # Dot at endpoints
        r = 3
        draw.ellipse([(x1-r, y1-r), (x1+r, y1+r)], fill=col)
        draw.ellipse([(x2-r, y2-r), (x2+r, y2+r)], fill=col)

        # Label at midpoint for longer candidates
        if c["length_ft"] >= 8:
            mx, my = c["midpoint"]
            mx *= scale; my *= scale
            txt = f"{c['length_ft']:.0f}ft/{c['angle_from_h']:.0f}°"
            # Small white background
            try:
                bb = draw.textbbox((mx, my), txt, font=font)
                draw.rectangle(bb, fill=(0, 0, 0, 128))
            except Exception:
                pass
            draw.text((mx, my), txt, fill=col, font=font)

    # Legend
    legend_y = 10
    legend_items = [
        ((255, 80, 80),   ">=20ft (structural)"),
        ((255,165,  0),   "10-20ft (probable)"),
        ((255,230,  0),   "7-10ft  (borderline)"),
        ((180,180,255),   "<7ft    (likely noise)"),
    ]
    draw.rectangle([(8, 6), (185, 10 + len(legend_items)*16 + 4)], fill=(20,20,20))
    for col, txt in legend_items:
        draw.rectangle([(12, legend_y), (28, legend_y+10)], fill=col)
        draw.text((32, legend_y), txt, fill=(220,220,220), font=font)
        legend_y += 15

    # Title
    title = f"{label}  ({len(candidates)} diagonal candidates ≥{RENDER_MIN_FT}ft)"
    draw.text((10, legend_y + 4), title, fill=(255,255,100), font=font)

    return img


def bucket_counts(candidates):
    """How many candidates survive each length cutoff?"""
    return {
        f">={b}ft": sum(1 for c in candidates if c["length_ft"] >= b)
        for b in REPORT_BUCKETS
    }


def length_histogram(candidates, bins=10):
    """Simple ASCII histogram of length distribution."""
    if not candidates:
        return "(none)"
    lengths = [c["length_ft"] for c in candidates]
    mn, mx  = min(lengths), max(lengths)
    if mx == mn:
        return f"all {mn:.1f}ft"
    step = (mx - mn) / bins
    buckets = [0] * bins
    for l in lengths:
        idx = min(int((l - mn) / step), bins - 1)
        buckets[idx] += 1
    lines = []
    for i, cnt in enumerate(buckets):
        lo = mn + i * step
        hi = lo + step
        bar = "#" * cnt
        lines.append(f"  {lo:5.1f}-{hi:5.1f}ft  {bar:<30} {cnt}")
    return "\n".join(lines)


def run():
    import re
    report_lines = []
    all_results  = {}

    def pr(s=""):
        print(s)
        report_lines.append(s)

    pr("=" * 72)
    pr("  GEOMETRY-BASED BRACE PROTOTYPE — Diagonal Candidate Report")
    pr("=" * 72)

    for (label, pdf_name, page_idx, scale_ratio) in TEST_PAGES:
        path = os.path.join(UPLOAD_DIR, pdf_name)
        if not os.path.exists(path):
            pr(f"\n[SKIP] {pdf_name} not found")
            continue

        ppf = scale_to_pts_per_foot(scale_ratio)
        doc = fitz.open(path)
        if page_idx >= len(doc):
            pr(f"\n[SKIP] {pdf_name} page {page_idx} out of range")
            doc.close()
            continue

        page   = doc[page_idx]
        if page.rotation in (90, 270):
            pw, ph = page.mediabox.width, page.mediabox.height
        else:
            pw, ph = page.rect.width, page.rect.height

        candidates = extract_diagonal_candidates(page, ppf, min_ft=RENDER_MIN_FT)
        buckets    = bucket_counts(candidates)

        pr(f"\n{'─'*72}")
        pr(f"  {label}")
        pr(f"  File : {pdf_name}  page={page_idx}  scale=1:{scale_ratio}  "
           f"({ppf:.1f} pts/ft)")
        pr(f"  Page : {pw:.0f} x {ph:.0f} pt")
        pr(f"  Total diagonal candidates (>={RENDER_MIN_FT}ft, 20-70°): {len(candidates)}")
        pr(f"  Survival by length cutoff: {buckets}")
        pr()

        # Length distribution
        pr("  Length distribution:")
        pr(length_histogram(candidates))
        pr()

        # Top candidates table
        pr(f"  {'#':>3}  {'length_ft':>9}  {'angle':>7}  {'stroke_w':>8}  "
           f"x1     y1     x2     y2")
        pr(f"  {'─'*3}  {'─'*9}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
        for i, c in enumerate(candidates[:40]):   # top 40
            pr(f"  {i+1:>3}  {c['length_ft']:>9.2f}  {c['angle_from_h']:>6.1f}°"
               f"  {c['stroke_w']:>8.2f}  "
               f"{c['x1']:>6.0f} {c['y1']:>6.0f} {c['x2']:>6.0f} {c['y2']:>6.0f}")
        if len(candidates) > 40:
            pr(f"  ... and {len(candidates)-40} more")

        # Stroke-width distribution for candidates ≥10ft
        long_cands = [c for c in candidates if c["length_ft"] >= 10]
        if long_cands:
            widths = sorted(set(round(c["stroke_w"], 2) for c in long_cands))
            pr(f"\n  Stroke widths on ≥10ft candidates: {widths}")

        all_results[label] = {
            "total_candidates": len(candidates),
            "buckets": buckets,
            "top10": [{"ft": c["length_ft"], "ang": c["angle_from_h"],
                       "sw": c["stroke_w"]} for c in candidates[:10]],
        }

        # Render overlay image
        img = render_overlay(page, candidates, ppf, label)
        out_path = os.path.join(OUT_DIR, f"{label}.png")
        img.save(out_path)
        pr(f"\n  Overlay saved: {out_path}")

        doc.close()

    # ── Summary across all pages ─────────────────────────────────────────────
    pr(f"\n{'='*72}")
    pr("  CROSS-PAGE SUMMARY")
    pr(f"{'='*72}")
    pr(f"\n  {'Label':<30} {'Total':>6}  " +
       "  ".join(f">={b}ft" for b in REPORT_BUCKETS))
    pr(f"  {'─'*30}  {'─'*6}  " + "  ".join("─"*6 for _ in REPORT_BUCKETS))
    for label, res in all_results.items():
        counts = "  ".join(f"{res['buckets'].get(f'>={b}ft',0):>6}" for b in REPORT_BUCKETS)
        pr(f"  {label:<30}  {res['total_candidates']:>6}  {counts}")

    pr(f"\n{'='*72}")
    pr("  INTERPRETATION GUIDE")
    pr(f"{'='*72}")
    pr("""
  RED  (>=20ft) — almost certainly a structural brace (bay-scale length)
  ORANGE (10-20ft) — probable brace or heavy diagonal kicker
  YELLOW (7-10ft)  — borderline; could be stair diagonal, kicker, or brace
  BLUE (<7ft)      — likely annotation, callout X, stair detail, or rafter tick

  Key question: do red/orange candidates cluster where braces are expected,
  or are they scattered across the whole page?

  Check overlay images in: """ + OUT_DIR)

    # Save JSON results
    json_path = os.path.join(OUT_DIR, "brace_proto_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    pr(f"\n  JSON results: {json_path}")

    # Save text report
    rpt_path = os.path.join(OUT_DIR, "brace_proto_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    pr(f"  Text report : {rpt_path}")


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    run()
