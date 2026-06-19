"""
Brace Overlay Accuracy Validation
----------------------------------
Selects 20+ HIGH confidence braces across multiple projects and drawing types.
For each candidate renders:
  1. Full-page overview with overlay
  2. Zoomed crop (150dpi) with endpoint markers

Alignment check targets:
  - Start point marker on brace endpoint
  - End point marker on brace endpoint
  - No overshoot / undershoot
  - No offset

Output: brace_overlay_validation/
  {idx}_{pdf}_{page}_{type}_overview.png
  {idx}_{pdf}_{page}_{type}_crop.png
  validation_manifest.json  (one record per candidate)
"""
import os, sys, io, math, json, re
import fitz
from PIL import Image, ImageDraw, ImageFont

def _save_img(img: Image.Image, path: str):
    """Save PIL image, using JPEG to avoid PIL 3.14 PNG tile-bounds bug."""
    jpg_path = path.replace(".png", ".jpg")
    img.convert("RGB").save(jpg_path, format="JPEG", quality=92)
    return jpg_path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUT_DIR    = os.path.join(HERE, "brace_overlay_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# Import production classifier functions (no hardcoded paths)
from brace_classifier import (
    extract_diagonals,
    classify_page_context,
    find_scale_annotations,
    classify,
    scale_to_pts_per_foot,
)

# ── PDFs to sample from (spread across projects + drawing types) ──────────────
# Format: (short_label, filename, scale_ratio)
SOURCES = [
    ("snaps",     "Structural snaps.pdf",                                       96.0),
    ("binder",    "#Structural binder.pdf",                                     64.0),
    ("07comb",    "07_STRUCTURAL_COMBINED.pdf",                                192.0),
    ("addendum",  "Latest_Structural dwg_Binder (Addendum-02).pdf",             96.0),
    ("s04",       "04_-_STRUCTURAL.pdf",                                        96.0),
    ("bayhealth", "2026.03.27_Bayhealth Sussex MOB_DD Set_Structural.pdf",     192.0),
    ("spruce",    "02 Struct 98 Spruce_2026-03-13_BID.pdf",                     96.0),
    ("martha",    "Pages from 2026.05.08_FF Martha Washington Building - Issued for Pricing.pdf", 96.0),
]

TARGET_TOTAL   = 25   # aim for 25 so we have margin
MAX_PER_SOURCE =  5   # cap per PDF to ensure project spread
MAX_PER_PAGE   =  2   # cap per page to ensure drawing-type spread


def get_font(size=11):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_overview(page, candidate, dpi=80):
    """
    Full-page render with overlay line + endpoint crosshairs.
    Uses the SAME coordinate transform as the production renderer:
      pixel = pdf_coord * (dpi / 72.0)
    """
    pix   = page.get_pixmap(dpi=dpi)
    img   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw  = ImageDraw.Draw(img)
    s     = dpi / 72.0

    c  = candidate
    x1 = c["x1"] * s;  y1 = c["y1"] * s
    x2 = c["x2"] * s;  y2 = c["y2"] * s

    # Overlay line (green, width 3)
    draw.line([(x1, y1), (x2, y2)], fill=(0, 230, 80), width=3)

    # Endpoint circles
    r = 5
    draw.ellipse([(x1-r, y1-r), (x1+r, y1+r)], outline=(255, 255, 0), width=2)
    draw.ellipse([(x2-r, y2-r), (x2+r, y2+r)], outline=(255, 80, 0), width=2)

    # Label
    font = get_font(10)
    txt  = f"{c['length_ft']:.1f}ft  {c['angle_from_h']:.0f}deg"
    mx   = (x1 + x2) / 2;  my = (y1 + y2) / 2
    draw.rectangle([(mx-1, my-1), (mx + len(txt)*6+2, my+13)], fill=(0, 0, 0))
    draw.text((mx, my), txt, fill=(0, 230, 80), font=font)
    return img


def render_crop(page, candidate, context_pt=300, dpi=150):
    """
    Zoomed crop centred on the candidate midpoint.
    Endpoint markers:
      Start (x1,y1) → yellow crosshair + 'S' label
      End   (x2,y2) → orange crosshair + 'E' label

    Transform: pixel = (pdf_coord - clip_origin) * (dpi / 72.0)
    """
    c  = candidate
    mx = (c["x1"] + c["x2"]) / 2
    my = (c["y1"] + c["y2"]) / 2

    half = max(context_pt,
               math.hypot(c["x2"] - c["x1"], c["y2"] - c["y1"]) * 0.7 + 100)
    pr   = page.rect
    clip = fitz.Rect(
        max(mx - half, pr.x0),
        max(my - half, pr.y0),
        min(mx + half, pr.x1),
        min(my + half, pr.y1),
    )

    # Guard: clip must have positive area
    if clip.width <= 1 or clip.height <= 1:
        clip = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y1)

    pix  = page.get_pixmap(dpi=dpi, clip=clip)
    if pix.width == 0 or pix.height == 0:
        raise ValueError(f"Empty pixmap for clip {clip} on page {page.number}")
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    s    = dpi / 72.0

    # Pixel coords of endpoints relative to clip
    px1 = (c["x1"] - clip.x0) * s;  py1 = (c["y1"] - clip.y0) * s
    px2 = (c["x2"] - clip.x0) * s;  py2 = (c["y2"] - clip.y0) * s

    # Overlay line
    draw.line([(px1, py1), (px2, py2)], fill=(0, 230, 80), width=4)

    # START crosshair (yellow)
    cr = 8
    draw.ellipse([(px1-cr, py1-cr), (px1+cr, py1+cr)],
                 outline=(255, 255, 0), width=3)
    draw.line([(px1-cr-4, py1), (px1+cr+4, py1)], fill=(255, 255, 0), width=2)
    draw.line([(px1, py1-cr-4), (px1, py1+cr+4)], fill=(255, 255, 0), width=2)

    # END crosshair (orange)
    draw.ellipse([(px2-cr, py2-cr), (px2+cr, py2+cr)],
                 outline=(255, 130, 0), width=3)
    draw.line([(px2-cr-4, py2), (px2+cr+4, py2)], fill=(255, 130, 0), width=2)
    draw.line([(px2, py2-cr-4), (px2, py2+cr+4)], fill=(255, 130, 0), width=2)

    font  = get_font(13)
    fontS = get_font(11)

    # S / E labels
    draw.text((px1 + cr + 3, py1 - 8), "S", fill=(255, 255, 0), font=font)
    draw.text((px2 + cr + 3, py2 - 8), "E", fill=(255, 130, 0), font=font)

    # Info bar at top
    info = (f"{c['length_ft']:.2f}ft  |  angle {c['angle_from_h']:.1f}deg  |  "
            f"sw={c['stroke_w']:.2f}pt  |  ctx={c.get('page_context','?')}")
    draw.rectangle([(0, 0), (img.width, 20)], fill=(10, 10, 10))
    draw.text((4, 3), info, fill=(0, 230, 80), font=fontS)

    # Coordinate values near endpoints
    coord_s = f"({c['x1']:.0f},{c['y1']:.0f})"
    coord_e = f"({c['x2']:.0f},{c['y2']:.0f})"
    draw.text((px1 + cr + 3, py1 + 4), coord_s, fill=(200, 200, 0), font=fontS)
    draw.text((px2 + cr + 3, py2 + 4), coord_e, fill=(200, 100, 0), font=fontS)

    return img


def collect_candidates():
    """
    Scan all sources, collect HIGH candidates spread across projects + contexts.
    Returns list of dicts with source info + candidate data.
    """
    all_candidates = []
    context_counts: dict[str, int] = defaultdict(int)

    for (label, pdf_name, scale_ratio) in SOURCES:
        path = os.path.join(UPLOAD_DIR, pdf_name)
        if not os.path.exists(path):
            print(f"  [SKIP] {pdf_name}")
            continue

        ppf    = scale_to_pts_per_foot(scale_ratio)
        doc    = fitz.open(path)
        source_candidates = []

        page_counts: dict[int, int] = defaultdict(int)

        for pg in range(len(doc)):
            page           = doc[pg]
            raw            = extract_diagonals(page, ppf)
            ctx, _         = classify_page_context(page)
            scale_anns     = find_scale_annotations(page)
            classified     = classify(raw, ctx, scale_anns, ppf)
            high           = [c for c in classified if c["confidence"] == "HIGH"]

            for c in high:
                if page_counts[pg] >= MAX_PER_PAGE:
                    break
                source_candidates.append({
                    "label":       label,
                    "pdf":         pdf_name,
                    "page":        pg,
                    "scale_ratio": scale_ratio,
                    "ppf":         ppf,
                    "context":     ctx,
                    **c,
                })
                page_counts[pg] += 1

        doc.close()

        # Sort by length desc, pick diverse set
        source_candidates.sort(key=lambda x: x["length_ft"], reverse=True)

        # Pick up to MAX_PER_SOURCE, preferring different contexts
        chosen = []
        seen_ctx = set()
        for c in source_candidates:
            if len(chosen) >= MAX_PER_SOURCE:
                break
            chosen.append(c)
            seen_ctx.add(c["context"])

        all_candidates.extend(chosen)
        print(f"  {label:<12} {pdf_name[:40]:<40}  "
              f"HIGH total={len(source_candidates)}  selected={len(chosen)}")

    return all_candidates


def run():
    print("=" * 76)
    print("  BRACE OVERLAY ACCURACY VALIDATION")
    print("=" * 76)
    print()

    candidates = collect_candidates()

    if len(candidates) < 20:
        print(f"\n  [WARN] Only {len(candidates)} candidates collected — need ≥20")
    else:
        print(f"\n  Collected {len(candidates)} candidates — proceeding with validation")

    manifest = []
    ctx_counter: dict[str, int] = defaultdict(int)
    proj_counter: dict[str, int] = defaultdict(int)

    print()
    print(f"  {'#':>3}  {'label':<10}  {'page':>4}  {'context':<25}  "
          f"{'length':>7}  {'angle':>6}  images")
    print(f"  {'─'*3}  {'─'*10}  {'─'*4}  {'─'*25}  "
          f"{'─'*7}  {'─'*6}  {'─'*40}")

    for idx, c in enumerate(candidates, 1):
        pdf_slug = re.sub(r'[^A-Za-z0-9]', '_', c["label"])
        ctx_slug = c["context"][:12].replace(" ", "_")
        base     = f"{idx:02d}_{pdf_slug}_p{c['page']:02d}_{ctx_slug}"

        # Re-open the page for rendering
        path = os.path.join(UPLOAD_DIR, c["pdf"])
        doc  = fitz.open(path)
        page = doc[c["page"]]

        render_ok = True
        try:
            # Render overview
            ov_img  = render_overview(page, c, dpi=80)
            ov_path = _save_img(ov_img, os.path.join(OUT_DIR, f"{base}_overview.png"))

            # Render crop
            cr_img  = render_crop(page, c, dpi=150)
            cr_path = _save_img(cr_img, os.path.join(OUT_DIR, f"{base}_crop.png"))
        except Exception as e:
            print(f"    [render error #{idx}: {e}]")
            render_ok = False
            ov_path = cr_path = "RENDER_ERROR"

        doc.close()

        manifest.append({
            "id":          idx,
            "label":       c["label"],
            "pdf":         c["pdf"],
            "page":        c["page"],
            "scale_ratio": c["scale_ratio"],
            "context":     c["context"],
            "length_ft":   c["length_ft"],
            "angle_from_h":c["angle_from_h"],
            "stroke_w":    c["stroke_w"],
            "x1": c["x1"], "y1": c["y1"],
            "x2": c["x2"], "y2": c["y2"],
            "overview":    ov_path,
            "crop":        cr_path,
            "alignment":   "PENDING",
        })

        ctx_counter[c["context"]]  += 1
        proj_counter[c["label"]]   += 1

        print(f"  {idx:>3}  {c['label']:<10}  {c['page']:>4}  "
              f"{c['context']:<25}  {c['length_ft']:>7.1f}  "
              f"{c['angle_from_h']:>6.1f}  {base}_crop.png")

    # Save manifest
    mf_path = os.path.join(OUT_DIR, "validation_manifest.json")
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print()
    print("  Context distribution:")
    for ctx, n in sorted(ctx_counter.items()):
        print(f"    {ctx:<30}: {n}")

    print()
    print("  Project distribution:")
    for proj, n in sorted(proj_counter.items()):
        print(f"    {proj:<15}: {n}")

    print()
    print(f"  Total candidates rendered: {len(candidates)}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Manifest   : {mf_path}")
    print("=" * 76)


if __name__ == "__main__":
    run()
