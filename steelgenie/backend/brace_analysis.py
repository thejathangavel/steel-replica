"""
Brace Availability Analysis
----------------------------
Scans our test PDF set for brace-related evidence WITHOUT running the full
extraction pipeline. Answers:
  1. Which PDFs contain brace-type labels?
  2. Plan sheet vs elevation sheet?
  3. What profile types (HSS, L, ISA, PIPE, etc.)?
  4. Label position relative to diagonal lines?
  5. How many placeholder braces does build_members already emit?

Run: python brace_analysis.py
"""
import os, sys, re, math
import fitz
import main

HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = main.UPLOAD_DIR

# ── Test cases (same as eval_harness) ────────────────────────────────────────
TEST_CASES = [
    {"name": "emerus_areaA",   "file": "1Pages from #Structural binder.pdf",  "page": 0,  "scale": 64.0},
    {"name": "areaB_skewed",   "file": "2Pages from #Structural binder.pdf",  "page": 0,  "scale": 64.0},
    {"name": "level2_framing", "file": "Structural snaps.pdf",                "page": 0,  "scale": 96.0},
    {"name": "snaps_skewed",   "file": "Structural snaps.pdf",                "page": 2,  "scale": 96.0},
    {"name": "roof_p41",       "file": "07_STRUCTURAL_COMBINED.pdf",          "page": 41, "scale": 192.0},
    {"name": "deflection_p14", "file": "07_STRUCTURAL_COMBINED.pdf",          "page": 14, "scale": 192.0},
]

# ── Broader set: scan ALL pages of structural PDFs for any brace labels ───────
SCAN_PDFS = [
    "#Structural binder.pdf",
    "Structural snaps.pdf",
    "07_STRUCTURAL_COMBINED.pdf",
    "Latest_Structural dwg_Binder (Addendum-02).pdf",
    "NCU SherMan_Structural.pdf",
    "04_-_STRUCTURAL.pdf",
    "2026.03.27_Bayhealth Sussex MOB_DD Set_Structural.pdf",
]

# ── Brace label patterns ──────────────────────────────────────────────────────
BRACE_PATTERNS = [
    (re.compile(r'\bISA[\dXx/]+', re.I),          "ISA"),
    (re.compile(r'\b2L\d+[Xx]\d+', re.I),         "2L (double-angle)"),
    (re.compile(r'\bL\d+[Xx]\d+[Xx][\d/]+', re.I),"L-angle (full)"),
    (re.compile(r'\bL\d+[Xx]\d+(?![Xx\d])', re.I),"L-angle (short)"),
    (re.compile(r'\bPIPE\s*\d', re.I),             "PIPE"),
    (re.compile(r'\bWT\d+[Xx]\d+', re.I),          "WT"),
    (re.compile(r'\bBRACE\b', re.I),               "BRACE (text)"),
    (re.compile(r'\bX-BRAC', re.I),                "X-BRACE (text)"),
    (re.compile(r'\bKBRACE\b|\bK-BRACE\b', re.I), "K-BRACE (text)"),
    # HSS on diagonal — need geometry to confirm, so flag all HSS for later
    (re.compile(r'\bHSS[\d.]+[Xx][\d.]+(?:[Xx][\d./]+)?', re.I), "HSS"),
]

def find_brace_labels_in_text(text_dict):
    """Return list of (label, pattern_type, x, y) from page text blocks."""
    hits = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if not t:
                    continue
                ox = span["origin"][0]
                oy = span["origin"][1]
                for pat, ptype in BRACE_PATTERNS:
                    for m in pat.finditer(t):
                        hits.append((m.group(), ptype, ox, oy))
    return hits


def get_diagonal_lines(page, plan_bounds, pts_per_foot, min_ft=4.0):
    """Extract diagonal solid lines (20-70° from H) from the page."""
    bx0, by0, bx1, by1 = plan_bounds
    min_pt = min_ft * pts_per_foot
    diags = []
    try:
        for d in page.get_drawings():
            dashes = str(d.get("dashes") or "").strip()
            da = re.match(r'\[([^\]]*)\]', dashes)
            if da and da.group(1).strip():
                continue
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
                    ang_h = min(ang, 180 - ang)
                    if not (20 <= ang_h <= 70):
                        continue
                    mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
                    # loose plan-bounds check
                    if not (bx0 - 200 <= mx <= bx1 + 200 and
                            by0 - 200 <= my <= by1 + 200):
                        continue
                    diags.append({
                        "x1": p1.x, "y1": p1.y,
                        "x2": p2.x, "y2": p2.y,
                        "length_pt": ln,
                        "length_ft": round(ln / pts_per_foot, 1),
                        "angle_from_h": round(ang_h, 1),
                        "width": d.get("width") or 0,
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return diags


def nearest_label_dist(lx, ly, brace_labels):
    """Return (dist_ft, label, type) of the closest brace label to point (lx, ly)."""
    best = None
    for (lb, ltype, bx, by) in brace_labels:
        d = math.hypot(lx - bx, ly - by)
        if best is None or d < best[0]:
            best = (d, lb, ltype)
    return best  # may be None


def label_position_relative_to_line(lbx, lby, x1, y1, x2, y2):
    """Where is the label relative to the line: midpoint / endpoint / off."""
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy) or 1.0
    t = max(0.0, min(1.0, ((lbx - x1) * dx + (lby - y1) * dy) / (ln * ln)))
    px = x1 + t * dx
    py = y1 + t * dy
    perp_dist = math.hypot(lbx - px, lby - py)
    if t < 0.2 or t > 0.8:
        zone = "near_endpoint"
    else:
        zone = "near_midpoint"
    return zone, round(perp_dist, 1), round(t, 2)


def count_build_members_braces(case):
    """Run build_members and count how many 'brace' type members it already emits."""
    path = os.path.join(UPLOAD_DIR, case["file"])
    if not os.path.exists(path):
        return None
    doc = fitz.open(path)
    page = doc[case["page"]]
    if page.rotation in (90, 270):
        pw, ph = page.mediabox.width, page.mediabox.height
    else:
        pw, ph = page.rect.width, page.rect.height
    td, is_raster = main._get_text_dict(page, pw, ph)
    if is_raster:
        return {"raster": True}
    pb = main.find_plan_boundary(page, pw, ph, text_dict=td)
    cs = main.detect_column_symbols(page)
    vg, hg = main.extract_grid_lines(page, pw, ph, pb, text_dict=td)
    profs = main.extract_profiles(page, pw, ph, pb, text_dict=td)
    ppf = main.scale_to_pts_per_foot(case["scale"])
    bdirs = main.detect_beam_directions(page, profs, pb)
    blm, alln = main.detect_beam_lines(page, profs, pb, pts_per_foot=ppf,
                                       column_symbols=cs, v_grid=vg, h_grid=hg)
    mem = main.build_members(profs, pw, ph, column_symbols=cs, v_grid=vg, h_grid=hg,
                             pts_per_foot=ppf, beam_dirs=bdirs, beam_line_map=blm,
                             plan_bounds=pb, is_vector=True)
    doc.close()
    brace_members = [m for m in mem if m.get("type") == "brace"]
    return {
        "total_members": len(mem),
        "brace_placeholders": len(brace_members),
        "brace_details": [{"profile": m["profile"],
                           "has_geometry": m.get("bx1") is not None,
                           "length_ft": m.get("length_ft", 0)} for m in brace_members],
        "all_profiles": [(p["profile"], main.classify_member(p["profile"])) for p in profs],
    }


def scan_pdf_for_braces(pdf_name, max_pages=80):
    """Scan ALL pages of a PDF for brace labels + diagonal geometry."""
    path = os.path.join(UPLOAD_DIR, pdf_name)
    if not os.path.exists(path):
        return None
    doc = fitz.open(path)
    results = []
    n = min(len(doc), max_pages)
    for pg_idx in range(n):
        page = doc[pg_idx]
        if page.rotation in (90, 270):
            pw, ph = page.mediabox.width, page.mediabox.height
        else:
            pw, ph = page.rect.width, page.rect.height
        try:
            td, is_raster = main._get_text_dict(page, pw, ph)
            if is_raster:
                continue
            pb = main.find_plan_boundary(page, pw, ph, text_dict=td)
            ppf = main.scale_to_pts_per_foot(96.0)  # assume 1/8"=1' as fallback
            labels = find_brace_labels_in_text(td)
            # Only flag pages that have brace-specific labels (not just HSS)
            brace_specific = [l for l in labels if l[1] not in ("HSS",)]
            diags = get_diagonal_lines(page, pb, ppf, min_ft=4.0) if labels else []
            # Page title hint
            page_text = page.get_text("text")[:200].replace("\n", " ")
            if labels or diags:
                results.append({
                    "page": pg_idx,
                    "page_hint": page_text[:120],
                    "brace_labels": labels,
                    "brace_specific": brace_specific,
                    "diagonal_lines": len(diags),
                    "diag_samples": diags[:5],
                })
        except Exception as e:
            continue
    doc.close()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN REPORT
# ══════════════════════════════════════════════════════════════════════════════
def main_run():
    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  BRACE AVAILABILITY ANALYSIS - SteelGenie Test Set")
    print(f"{SEP}\n")

    # ── PART 1: build_members placeholder count on eval cases ────────────────
    print("-- PART 1: Placeholder braces from build_members (eval cases) ------\n")
    for case in TEST_CASES:
        res = count_build_members_braces(case)
        if res is None:
            print(f"  {case['name']:20s} FILE NOT FOUND")
            continue
        if res.get("raster"):
            print(f"  {case['name']:20s} RASTER — skipped")
            continue
        bp = res["brace_placeholders"]
        print(f"  {case['name']:20s}  {bp} placeholder brace(s)  "
              f"(total members={res['total_members']})")
        if bp:
            for d in res["brace_details"]:
                geo = "WITH geometry" if d["has_geometry"] else "NO geometry (point only)"
                print(f"    → {d['profile']:16s}  {geo}  len={d['length_ft']}ft")
        # Also show any brace-related profiles in the raw extract
        brace_profs = [(p, c) for p, c in res["all_profiles"] if c == "brace"]
        if brace_profs:
            print(f"    Raw brace-type profiles: {brace_profs}")

    # ── PART 2: Full PDF scan ─────────────────────────────────────────────────
    print(f"\n-- PART 2: Full PDF scan for brace labels + diagonal geometry -------\n")
    for pdf_name in SCAN_PDFS:
        path = os.path.join(UPLOAD_DIR, pdf_name)
        if not os.path.exists(path):
            print(f"  {pdf_name[:50]:50s}  NOT FOUND")
            continue
        results = scan_pdf_for_braces(pdf_name)
        pages_with_labels = [r for r in results if r["brace_labels"]]
        pages_specific    = [r for r in results if r["brace_specific"]]
        print(f"\n  PDF: {pdf_name}")
        print(f"  Pages scanned: {len(results) + (fitz.open(os.path.join(UPLOAD_DIR, pdf_name)).page_count - len(results))}")
        print(f"  Pages with ANY brace labels: {len(pages_with_labels)}")
        print(f"  Pages with NON-HSS brace labels (L/ISA/BRACE/WT): {len(pages_specific)}")

        for r in pages_with_labels[:6]:  # show up to 6 pages
            label_summary = {}
            for (lb, ltype, lx, ly) in r["brace_labels"]:
                label_summary.setdefault(ltype, []).append(lb)
            print(f"\n    Page {r['page']:3d} | diag lines: {r['diagonal_lines']:3d} | "
                  f"hint: {r['page_hint'][:60]}")
            for ltype, examples in label_summary.items():
                unique = list(dict.fromkeys(examples))[:4]
                print(f"      {ltype:25s}: {unique}")

            # For non-HSS brace pages, show label→diagonal proximity
            if r["brace_specific"] and r["diag_samples"]:
                print(f"      Diagonal line samples (angle / length):")
                for dg in r["diag_samples"][:3]:
                    print(f"        {dg['angle_from_h']:5.1f}° from H  "
                          f"len={dg['length_ft']:5.1f}ft  width={dg['width']:.2f}pt")
                # check label→line proximity for first non-HSS label
                first_lbl = r["brace_specific"][0]
                for dg in r["diag_samples"][:3]:
                    # check proximity to both endpoints and midpoint
                    mx = (dg["x1"] + dg["x2"]) / 2
                    my = (dg["y1"] + dg["y2"]) / 2
                    zone, perp, t = label_position_relative_to_line(
                        first_lbl[2], first_lbl[3],
                        dg["x1"], dg["y1"], dg["x2"], dg["y2"])
                    ppf_fallback = 9.0  # ~1/8"=1' in pts
                    dist_ft = round(perp / ppf_fallback, 1)
                    print(f"        Label '{first_lbl[0]}' → line: "
                          f"perp={perp:.0f}pt (~{dist_ft}ft)  t={t}  zone={zone}")

    print(f"\n{SEP}")
    print("  END OF ANALYSIS")
    print(f"{SEP}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main_run()
