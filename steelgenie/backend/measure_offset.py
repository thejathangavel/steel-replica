"""
measure_offset.py
-----------------
Measures how far each beam overlay is from the nearest actual drawn line.

For every beam member returned by /analyse, find the closest parallel
drawn line in the PDF (using collect_line_widths) and report the
perpendicular distance (offset) in points and inches.

Run:  python measure_offset.py
"""

import math, sys, requests, fitz

PDF = "Structural snaps.pdf"
UPLOAD_DIR = r"D:\Steel-ghost\steelgenie\backend\uploads"
SCALE = 96   # 1/8" = 1'-0"
BASE  = "http://localhost:8000"

# ── 1. get members from API ───────────────────────────────────────────────────
r = requests.post(f"{BASE}/analyse",
                  json={"filename": PDF, "page_index": 0, "scale_ratio": SCALE},
                  timeout=120)
data  = r.json()
members = data["members"]
beams = [m for m in members if m["type"] == "beam" and m.get("bx1") is not None]
print(f"Beams to measure: {len(beams)}")

# ── 2. open PDF and collect all drawn lines ───────────────────────────────────
doc  = fitz.open(f"{UPLOAD_DIR}\\{PDF}")
page = doc[0]
pw, ph = page.rect.width, page.rect.height

lines = []
for d in page.get_drawings():
    wv = float(d.get("width") or 0.5)
    for it in d.get("items", []):
        if it[0] != "l":
            continue
        p1, p2 = it[1], it[2]
        ln = math.hypot(p2.x - p1.x, p2.y - p1.y)
        if ln < 20:
            continue
        lines.append((p1.x, p1.y, p2.x, p2.y, ln, wv))

print(f"PDF lines collected: {len(lines)}")

# ── 3. for each beam measure perpendicular offset to nearest parallel line ────
def measure_beam(m):
    x1 = m["bx1"] * pw;  y1 = m["by1"] * ph
    x2 = m["bx2"] * pw;  y2 = m["by2"] * ph
    bln = math.hypot(x2-x1, y2-y1)
    if bln < 1:
        return None
    ux, uy = (x2-x1)/bln, (y2-y1)/bln
    px, py = -uy, ux          # perpendicular unit vector

    best_perp = None
    best_w    = 0.0
    best_all  = None          # nearest regardless of thickness

    for (qx1,qy1,qx2,qy2,qln,qw) in lines:
        if qln < 0.3 * bln:  # must cover at least 30% of beam
            continue
        qdx, qdy = qx2-qx1, qy2-qy1
        dot = abs((qdx*ux + qdy*uy) / qln)
        if dot < 0.94:        # not parallel enough
            continue
        perp = abs((qx1-x1)*px + (qy1-y1)*py)
        if perp > 60:         # >5 inches — definitely a different member
            continue
        # nearest thick line (beam lines are thick)
        if qw >= 0.8 and (best_perp is None or perp < best_perp):
            best_perp = perp
            best_w    = qw
        # nearest line of any width
        if best_all is None or perp < best_all[0]:
            best_all = (perp, qw)

    return best_perp, best_w, best_all

# ── 4. report ─────────────────────────────────────────────────────────────────
PTS_PER_INCH = 72.0

results = []
for m in beams:
    res = measure_beam(m)
    if res is None:
        continue
    thick_perp, thick_w, any_nearest = res
    # offset from thick line (what center_beam_overlays targets)
    off_thick = thick_perp if thick_perp is not None else None
    # offset from absolute nearest line (any width)
    off_any   = any_nearest[0] if any_nearest else None
    results.append((m["profile"], m.get("beam_dir","?"), round(m["length_ft"],1),
                    off_thick, thick_w, off_any))

# sort by thick-line offset descending (worst first)
results.sort(key=lambda r: (r[3] or 999), reverse=True)

print(f"\n{'Profile':<14} {'Dir'} {'Len':>6}  {'Offset-thick(pt)':>17}  {'(in)':>6}  {'Offset-any(pt)':>14}  {'ThickW':>7}")
print("-"*80)
worst_thick = []
for prof, bdir, lft, ot, tw, oa in results:
    ot_str = f"{ot:6.1f}" if ot is not None else "  none"
    ot_in  = f"{ot/PTS_PER_INCH:.3f}" if ot is not None else "  n/a"
    oa_str = f"{oa:6.1f}" if oa is not None else "  none"
    print(f"{prof:<14} {bdir}  {lft:>6.1f}  {ot_str:>17}  {ot_in:>6}  {oa_str:>14}  {tw:>7.2f}")
    if ot is not None and ot > 4.0:
        worst_thick.append((prof, bdir, lft, ot, ot/PTS_PER_INCH))

print(f"\n── SUMMARY ──────────────────────────────────────────")
off_vals = [r[3] for r in results if r[3] is not None]
if off_vals:
    print(f"  Beams measured       : {len(off_vals)}")
    print(f"  Mean offset (thick)  : {sum(off_vals)/len(off_vals):.2f} pt  ({sum(off_vals)/len(off_vals)/PTS_PER_INCH:.3f} in)")
    print(f"  Max  offset (thick)  : {max(off_vals):.2f} pt  ({max(off_vals)/PTS_PER_INCH:.3f} in)")
    print(f"  Offset ≤ 2 pt        : {sum(1 for v in off_vals if v<=2)} / {len(off_vals)}")
    print(f"  Offset ≤ 5 pt        : {sum(1 for v in off_vals if v<=5)} / {len(off_vals)}")
    print(f"  Offset > 5 pt (bad)  : {sum(1 for v in off_vals if v>5)} beams")
print(f"\n── WORST OFFSETS (>4 pt from thick line) ────────────")
for prof, bdir, lft, ot, ot_in in sorted(worst_thick, key=lambda x: -x[3])[:15]:
    print(f"  {prof:<14} {bdir}  {lft:>6.1f} ft   {ot:5.1f} pt = {ot_in:.3f} in")
