"""
False-Positive Study — HIGH Confidence Classifier
---------------------------------------------------
Selects 30 HIGH confidence candidates:
  • 10 from brace elevation sheets
  • 10 from framing plan sheets
  • 10 from roof plan sheets

For each candidate renders:
  1. A full-page overview overlay (green candidate highlighted)
  2. A zoomed-in crop centred on the candidate (~300pt context radius)

Usage:
  python brace_fp_study.py
"""
import os, sys, io, math, json
import fitz
from PIL import Image, ImageDraw, ImageFont

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUT_DIR    = os.path.join(HERE, "brace_fp_study_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Scale helper ──────────────────────────────────────────────────────────────
def ppf(scale): return 864.0 / scale if scale > 0 else 9.0

# ── Candidate extraction (same logic as brace_classifier.py) ─────────────────
import re as _re

def extract_diagonals(page, pts_per_foot, min_ft=3.0):
    min_pt = min_ft * pts_per_foot
    results, seen = [], set()
    for d in page.get_drawings():
        dashes = str(d.get("dashes") or "").strip()
        da = _re.match(r'\[([^\]]*)\]', dashes)
        if da and da.group(1).strip():
            continue
        sw = float(d.get("width") or 0)
        for item in d.get("items", []):
            if item[0] != "l":
                continue
            try:
                p1, p2 = item[1], item[2]
                dx, dy = p2.x - p1.x, p2.y - p1.y
                ln = math.hypot(dx, dy)
                if ln < min_pt:
                    continue
                ang = abs(math.degrees(math.atan2(dy, dx))) % 180
                ah  = min(ang, 180 - ang)
                if not (20.0 <= ah <= 70.0):
                    continue
                key = (round(p1.x), round(p1.y), round(p2.x), round(p2.y))
                rk  = (round(p2.x), round(p2.y), round(p1.x), round(p1.y))
                if key in seen or rk in seen:
                    continue
                seen.add(key)
                results.append({
                    "x1": p1.x, "y1": p1.y,
                    "x2": p2.x, "y2": p2.y,
                    "length_ft": round(ln / pts_per_foot, 2),
                    "angle":     round(ah, 1),
                    "stroke_w":  sw,
                })
            except Exception:
                continue
    results.sort(key=lambda c: c["length_ft"], reverse=True)
    return results

def find_density_clusters(candidates, bin_ft=1.0, threshold=8):
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, c in enumerate(candidates):
        b = int(c["length_ft"] / bin_ft)
        buckets[b].append(i)
    cluster = set()
    for idx_list in buckets.values():
        if len(idx_list) >= threshold:
            cluster.update(idx_list)
    return cluster

def classify_candidates(candidates):
    cluster = find_density_clusters(candidates)
    out = []
    for i, c in enumerate(candidates):
        c = dict(c)
        if i in cluster:
            c["confidence"] = "REJECT"; c["reason"] = "density_cluster"
        elif c["length_ft"] < 8.0:
            c["confidence"] = "REJECT"; c["reason"] = "too_short"
        elif c["length_ft"] < 15.0:
            c["confidence"] = "MEDIUM"; c["reason"] = None
        else:
            c["confidence"] = "HIGH";   c["reason"] = None
        out.append(c)
    return out

# ── Pages to sample (manually curated by sheet type) ─────────────────────────
#
# Format: (group_label, pdf, page_idx, scale_ratio, sheet_type_hint)
#
# Elevation sheets — pages confirmed/likely to be braced frame elevations
ELEVATION_PAGES = [
    ("snaps",   "Structural snaps.pdf",          6,  96.0),
    ("snaps",   "Structural snaps.pdf",          7,  96.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",    47, 192.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",    48, 192.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",    49, 192.0),
]

# Framing plan sheets — floor / structural framing plan pages
FRAMING_PAGES = [
    ("snaps",   "Structural snaps.pdf",           2,  96.0),
    ("snaps",   "Structural snaps.pdf",           5,  96.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",     8, 192.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",     9, 192.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",    13, 192.0),
    ("binder",  "#Structural binder.pdf",        12,  64.0),
    ("binder",  "#Structural binder.pdf",        16,  64.0),
]

# Roof plan sheets — roof framing / roof deck plan pages
ROOF_PAGES = [
    ("snaps",   "Structural snaps.pdf",           3,  96.0),
    ("binder",  "#Structural binder.pdf",        10,  64.0),
    ("binder",  "#Structural binder.pdf",        11,  64.0),
    ("07comb",  "07_STRUCTURAL_COMBINED.pdf",    20, 192.0),
    ("latest",  "Latest_Structural dwg_Binder (Addendum-02).pdf", 1, 96.0),
]

GROUPS = [
    ("A_elevation",  ELEVATION_PAGES),
    ("B_framing",    FRAMING_PAGES),
    ("C_roof",       ROOF_PAGES),
]

# ── Rendering helpers ─────────────────────────────────────────────────────────
def get_font(size=11):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(name, size)
        except: pass
    return ImageFont.load_default()

def render_overview(page, candidate, dpi=72):
    """Full-page render with candidate highlighted in bright green."""
    pix   = page.get_pixmap(dpi=dpi)
    img   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw  = ImageDraw.Draw(img)
    s     = dpi / 72.0
    c     = candidate
    x1, y1, x2, y2 = c["x1"]*s, c["y1"]*s, c["x2"]*s, c["y2"]*s
    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 80), width=4)
    r = 5
    for px_, py_ in [(x1, y1), (x2, y2)]:
        draw.ellipse([(px_-r, py_-r), (px_+r, py_+r)], fill=(0, 255, 80))
    font = get_font(12)
    info = f"{c['length_ft']:.1f}ft  {c['angle']:.0f}deg  sw={c['stroke_w']:.2f}pt"
    mx, my = (x1+x2)/2, (y1+y2)/2
    draw.rectangle([(mx-2, my-2), (mx+len(info)*7, my+14)], fill=(0,0,0))
    draw.text((mx, my), info, fill=(0,255,80), font=font)
    return img

def render_crop(page, candidate, context_pt=280, dpi=150):
    """Zoomed crop around the candidate midpoint."""
    c  = candidate
    mx = (c["x1"] + c["x2"]) / 2
    my = (c["y1"] + c["y2"]) / 2
    # Enlarge context to fit the whole segment
    half = max(context_pt,
               math.hypot(c["x2"]-c["x1"], c["y2"]-c["y1"]) * 0.6 + 80)
    clip = fitz.Rect(mx - half, my - half, mx + half, my + half)
    pix  = page.get_pixmap(dpi=dpi, clip=clip)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    s    = dpi / 72.0
    # Candidate coords relative to clip
    x1 = (c["x1"] - clip.x0) * s
    y1 = (c["y1"] - clip.y0) * s
    x2 = (c["x2"] - clip.x0) * s
    y2 = (c["y2"] - clip.y0) * s
    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 80), width=5)
    r = 6
    for px_, py_ in [(x1, y1), (x2, y2)]:
        draw.ellipse([(px_-r, py_-r), (px_+r, py_+r)], fill=(255, 80, 0))
    font = get_font(13)
    info = f"{c['length_ft']:.1f}ft | {c['angle']:.0f}deg | sw={c['stroke_w']:.2f}pt"
    draw.rectangle([(4, 4), (len(info)*8+8, 22)], fill=(0,0,0))
    draw.text((6, 5), info, fill=(0, 255, 80), font=font)
    return img

# ── Collect N highest-confidence candidates from a list of pages ──────────────
def collect_high_from_pages(page_list, n_candidates=10):
    """
    Open each page, classify candidates, return the top n HIGH candidates
    with their source info (pdf, page_idx, candidate_dict).
    """
    all_high = []
    for (tag, pdf_name, pg_idx, scale) in page_list:
        path = os.path.join(UPLOAD_DIR, pdf_name)
        if not os.path.exists(path):
            print(f"  [SKIP] {pdf_name} not found")
            continue
        try:
            doc  = fitz.open(path)
            if pg_idx >= len(doc):
                doc.close()
                continue
            page = doc[pg_idx]
            pts  = ppf(scale)
            raw  = extract_diagonals(page, pts, min_ft=8.0)
            clas = classify_candidates(raw)
            high = [c for c in clas if c["confidence"] == "HIGH"]
            # Deduplicate by length bucket to get varied samples
            seen_buckets = set()
            unique_high  = []
            for h in high:
                bk = round(h["length_ft"] / 2) * 2  # 2ft bucket
                if bk not in seen_buckets:
                    seen_buckets.add(bk)
                    unique_high.append(h)
            for h in unique_high:
                all_high.append((tag, pdf_name, pg_idx, scale, page, h, doc))
        except Exception as e:
            print(f"  [ERR] {pdf_name} p{pg_idx}: {e}")

    # Sort by length descending, pick n
    all_high.sort(key=lambda x: x[5]["length_ft"], reverse=True)
    # Spread across pages — pick round-robin to avoid all from one page
    from collections import defaultdict
    by_page = defaultdict(list)
    for item in all_high:
        key = (item[1], item[2])
        by_page[key].append(item)

    selected = []
    round_idx = 0
    page_keys = list(by_page.keys())
    while len(selected) < n_candidates:
        added_any = False
        for pk in page_keys:
            if round_idx < len(by_page[pk]):
                selected.append(by_page[pk][round_idx])
                added_any = True
                if len(selected) >= n_candidates:
                    break
        if not added_any:
            break
        round_idx += 1

    return selected[:n_candidates]

# ── Main study ────────────────────────────────────────────────────────────────
def run():
    report = []
    def pr(s=""):
        print(s)
        report.append(s)

    pr("=" * 76)
    pr("  FALSE-POSITIVE STUDY — HIGH Confidence Brace Classifier")
    pr("=" * 76)

    study_data = []
    candidate_counter = 0

    for (group_name, page_list) in GROUPS:
        pr(f"\n{'─'*76}")
        pr(f"  GROUP {group_name}")
        pr(f"{'─'*76}")

        selected = collect_high_from_pages(page_list, n_candidates=10)
        if not selected:
            pr("  [No HIGH candidates found in this group]")
            continue

        # Keep docs open only as long as we need them for rendering
        open_docs = {}  # path -> doc

        for (tag, pdf_name, pg_idx, scale, page_ref, cand, doc_ref) in selected:
            candidate_counter += 1
            label = f"{group_name}_{candidate_counter:02d}"
            pdf_short = pdf_name.replace("#", "").replace(" ", "_")[:25]
            item_label = f"{label}  [{pdf_short} p{pg_idx}]"

            pr(f"\n  {item_label}")
            pr(f"    length  : {cand['length_ft']:.2f} ft")
            pr(f"    angle   : {cand['angle']:.1f} deg")
            pr(f"    stroke  : {cand['stroke_w']:.2f} pt")
            pr(f"    coords  : ({cand['x1']:.0f},{cand['y1']:.0f}) → "
               f"({cand['x2']:.0f},{cand['y2']:.0f})")
            pr(f"    scale   : 1:{scale:.0f} ({864/scale:.2f} pt/ft)")
            pr(f"    [MANUAL CLASSIFICATION]: ___________________________")

            # Render images
            try:
                # Overview
                overview = render_overview(page_ref, cand, dpi=72)
                ov_path  = os.path.join(OUT_DIR, f"{label}_overview.png")
                overview.save(ov_path)

                # Crop
                crop     = render_crop(page_ref, cand, dpi=150)
                cr_path  = os.path.join(OUT_DIR, f"{label}_crop.png")
                crop.save(cr_path)

                pr(f"    overview: {ov_path}")
                pr(f"    crop    : {cr_path}")
            except Exception as e:
                pr(f"    [render error: {e}]")

            study_data.append({
                "id": label,
                "group": group_name,
                "pdf": pdf_name,
                "page": pg_idx,
                "scale": scale,
                "length_ft": cand["length_ft"],
                "angle": cand["angle"],
                "stroke_w": cand["stroke_w"],
                "coords": [cand["x1"], cand["y1"], cand["x2"], cand["y2"]],
                "classification": "PENDING",
            })

        # Close all docs opened in this group
        for (_, _, _, _, page_ref, _, doc_ref) in selected:
            try:
                doc_ref.close()
            except Exception:
                pass

    # Save JSON study sheet
    json_path = os.path.join(OUT_DIR, "fp_study_sheet.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(study_data, f, indent=2)

    pr(f"\n{'='*76}")
    pr(f"  Study data: {json_path}")
    pr(f"  Images in : {OUT_DIR}")
    pr(f"  Total candidates rendered: {candidate_counter}")
    pr("=" * 76)

    # Save report
    rpt_path = os.path.join(OUT_DIR, "fp_study_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    pr(f"  Report    : {rpt_path}")


if __name__ == "__main__":
    run()
