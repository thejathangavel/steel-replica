"""
Steel-Genie extraction EVAL HARNESS.

Runs the full extraction pipeline over a registry of test drawings and reports
QUALITY METRICS per sheet, then compares to a saved baseline so any change is
*measured* — no more "looked fine on the one drawing I tried".

Metrics per sheet
-----------------
  beams / columns / joists   : member counts (regression signal)
  phantom                    : beam overlays NOT backed by any drawn line
                               (drawn on empty space) — a false positive
  overshoot                  : beams extending > 3 ft past the outermost
                               drawn line on their axis
  duplicate                  : parallel, coincident, overlapping beam pairs
  elapsed_s                  : wall-clock pipeline time

Usage
-----
  python eval_harness.py              # run, print table, compare to baseline
  python eval_harness.py --save       # run and OVERWRITE the baseline
  python eval_harness.py --add "<file>" <page> <scale> "<name>"   # register a case

Ground truth (optional): put true counts in eval_groundtruth.json keyed by case
name -> {"beams": N, "columns": N}; the harness then also reports recall.
"""
import os, sys, json, math, time
import fitz
import main

HERE        = os.path.dirname(os.path.abspath(__file__))
REGISTRY    = os.path.join(HERE, "eval_cases.json")
BASELINE    = os.path.join(HERE, "eval_baseline.json")
GROUNDTRUTH = os.path.join(HERE, "eval_groundtruth.json")

# Default registry — edit eval_cases.json (created on first run) to curate.
DEFAULT_CASES = [
    {"name": "emerus_areaA",  "file": "1Pages from #Structural binder.pdf", "page": 0,  "scale": 64.0,  "unlabeled": True},
    {"name": "areaB_skewed",  "file": "2Pages from #Structural binder.pdf", "page": 0,  "scale": 64.0,  "unlabeled": True},
    {"name": "level2_framing", "file": "Structural snaps.pdf",               "page": 0,  "scale": 96.0,  "unlabeled": False},
    {"name": "snaps_skewed",   "file": "Structural snaps.pdf",               "page": 2,  "scale": 96.0,  "unlabeled": True},
    {"name": "roof_p41",       "file": "07_STRUCTURAL_COMBINED.pdf",         "page": 41, "scale": 192.0, "unlabeled": False},
    {"name": "deflection_p14", "file": "07_STRUCTURAL_COMBINED.pdf",         "page": 14, "scale": 192.0, "unlabeled": True},
]


def _run_pipeline(case):
    """Run the full member pipeline (unrotated space) and return members + ctx."""
    path = os.path.join(main.UPLOAD_DIR, case["file"])
    if not os.path.exists(path):
        return None
    doc = fitz.open(path)
    page = doc[case["page"]]
    pw, ph = ((page.mediabox.width, page.mediabox.height)
              if page.rotation in (90, 270) else (page.rect.width, page.rect.height))
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
    ccx, ccy = main.clean_column_lines(vg, hg, cs)
    mem = main.trim_beam_overshoot(mem, pw, ph, ppf, col_x=ccx, col_y=ccy)
    mem = main.emit_symbol_columns(mem, cs, vg, hg, pw, ph)
    if case.get("unlabeled"):
        mem = main.add_unlabeled_lines_universal(mem, alln, pb, pw, ph, ppf,
                                                 v_grid=vg, h_grid=hg, column_symbols=cs)
    mem = main.snap_beam_ends_to_supports(mem, alln, ccx, ccy, pw, ph, ppf)
    lw = main.collect_line_widths(page, pb)
    mem = main.center_beam_overlays(mem, lw, pw, ph)
    if hasattr(main, "dedup_overlapping_beams"):
        mem = main.dedup_overlapping_beams(mem, pw, ph, ppf)
    if hasattr(main, "trim_floating_endpoints"):
        mem = main.trim_floating_endpoints(mem, pw, ph, ppf)
    return {"members": mem, "pw": pw, "ph": ph, "ppf": ppf,
            "lines_w": lw, "ccx": ccx, "ccy": ccy}


def _metrics(ctx):
    mem, pw, ph, ppf = ctx["members"], ctx["pw"], ctx["ph"], ctx["ppf"]
    lw = ctx["lines_w"]; ccx, ccy = ctx["ccx"], ctx["ccy"]
    beams = [m for m in mem if m.get("type") == "beam" and m.get("bx1") is not None]
    cols  = sum(1 for m in mem if m.get("type") == "column")
    joists = sum(1 for m in mem if m.get("type") == "joist")

    # FLOATING endpoint: a beam end that lands on neither a column NOR a
    # perpendicular beam — i.e. it overshoots into empty space (the real
    # phantom/overshoot signal; "on a grid/dim line" does NOT count as support).
    colpts = [(m["x"]*pw, m["y"]*ph) for m in mem if m.get("type") == "column"]
    R = 1.8 * ppf
    def supported(ex, ey, sm):
        for cx, cy in colpts:
            if math.hypot(ex-cx, ey-cy) < R:
                return True
        sdx, sdy = (sm["bx2"]-sm["bx1"])*pw, (sm["by2"]-sm["by1"])*ph
        sl = math.hypot(sdx, sdy) or 1.0; sux, suy = sdx/sl, sdy/sl
        for n in beams:
            if n is sm:
                continue
            nx1, ny1, nx2, ny2 = n["bx1"]*pw, n["by1"]*ph, n["bx2"]*pw, n["by2"]*ph
            ndx, ndy = nx2-nx1, ny2-ny1; nl = math.hypot(ndx, ndy) or 1.0
            if abs(sux*(ndx/nl)+suy*(ndy/nl)) > 0.5:
                continue                            # need a perpendicular beam
            pr = max(0.0, min(1.0, ((ex-nx1)*ndx+(ey-ny1)*ndy)/(nl*nl)))
            if math.hypot(ex-(nx1+pr*ndx), ey-(ny1+pr*ndy)) < R:
                return True
        return False
    floating = 0
    for m in beams:
        s1 = supported(m["bx1"]*pw, m["by1"]*ph, m)
        s2 = supported(m["bx2"]*pw, m["by2"]*ph, m)
        if not s1 or not s2:
            floating += 1

    # phantom: < 55% of the beam length is within 6pt of a parallel drawn line
    def coverage(m):
        x1, y1, x2, y2 = m["bx1"]*pw, m["by1"]*ph, m["bx2"]*pw, m["by2"]*ph
        L = math.hypot(x2-x1, y2-y1) or 1.0
        ux, uy = (x2-x1)/L, (y2-y1)/L
        N = max(6, int(L/6)); hit = 0
        for k in range(N+1):
            t = L*k/N; sx, sy = x1+ux*t, y1+uy*t
            for (qx1, qy1, qx2, qy2, qln, qw) in lw:
                qdx, qdy = qx2-qx1, qy2-qy1; ql = math.hypot(qdx, qdy) or 1.0
                if abs((qdx*ux+qdy*uy)/ql) < 0.97:
                    continue
                pr = max(0.0, min(1.0, ((sx-qx1)*qdx+(sy-qy1)*qdy)/(ql*ql)))
                if math.hypot(sx-(qx1+pr*qdx), sy-(qy1+pr*qdy)) < 6:
                    hit += 1; break
        return hit/(N+1)

    # overshoot: beam end extends > 3 ft past the outermost column on its axis
    def overshoots(m):
        x1, y1, x2, y2 = m["bx1"]*pw, m["by1"]*ph, m["bx2"]*pw, m["by2"]*ph
        adx, ady = abs(x2-x1), abs(y2-y1)
        if adx >= ady:   # H beam — bounded by column X lines
            if not ccx: return False
            lo, hi = min(x1, x2), max(x1, x2)
            return (lo < min(ccx) - 3*ppf) or (hi > max(ccx) + 3*ppf)
        else:
            if not ccy: return False
            lo, hi = min(y1, y2), max(y1, y2)
            return (lo < min(ccy) - 3*ppf) or (hi > max(ccy) + 3*ppf)

    phantom = sum(1 for m in beams if coverage(m) < 0.55)
    overshoot = sum(1 for m in beams if overshoots(m))

    # duplicate pairs: parallel, coincident (<8px perp), overlap >=50%
    dup = 0
    for i in range(len(beams)):
        ax1, ay1, ax2, ay2 = (beams[i]["bx1"]*pw, beams[i]["by1"]*ph,
                              beams[i]["bx2"]*pw, beams[i]["by2"]*ph)
        La = math.hypot(ax2-ax1, ay2-ay1) or 1.0
        ux, uy = (ax2-ax1)/La, (ay2-ay1)/La; px, py = -uy, ux
        for j in range(i+1, len(beams)):
            bx1, by1, bx2, by2 = (beams[j]["bx1"]*pw, beams[j]["by1"]*ph,
                                  beams[j]["bx2"]*pw, beams[j]["by2"]*ph)
            Lb = math.hypot(bx2-bx1, by2-by1) or 1.0
            if abs(((bx2-bx1)*ux+(by2-by1)*uy)/Lb) < 0.96: continue
            if abs((bx1-ax1)*px+(by1-ay1)*py) > 8: continue
            t1 = (bx1-ax1)*ux+(by1-ay1)*uy; t2 = (bx2-ax1)*ux+(by2-ay1)*uy
            if min(La, max(t1, t2)) - max(0.0, min(t1, t2)) > 0.5*min(La, Lb):
                dup += 1
    return {"beams": len(beams), "columns": cols, "joists": joists,
            "floating": floating, "phantom": phantom, "overshoot": overshoot,
            "duplicate": dup}


def main_run():
    save = "--save" in sys.argv
    cases = DEFAULT_CASES
    if os.path.exists(REGISTRY):
        try: cases = json.load(open(REGISTRY))
        except Exception: pass
    else:
        json.dump(DEFAULT_CASES, open(REGISTRY, "w"), indent=2)

    gt = {}
    if os.path.exists(GROUNDTRUTH):
        try: gt = json.load(open(GROUNDTRUTH))
        except Exception: pass
    base = {}
    if os.path.exists(BASELINE):
        try: base = json.load(open(BASELINE))
        except Exception: pass

    results = {}
    print(f"\n{'case':16} {'beams':>6} {'joist':>6} {'cols':>5} {'float':>6} "
          f"{'dup':>4} {'sec':>5}  vs-baseline")
    print("-" * 92)
    for c in cases:
        t0 = time.time()
        ctx = _run_pipeline(c)
        dt = round(time.time()-t0, 1)
        if ctx is None:
            print(f"{c['name']:16} FILE NOT FOUND"); continue
        if ctx.get("raster"):
            print(f"{c['name']:16} RASTER (skipped)"); continue
        m = _metrics(ctx); m["elapsed_s"] = dt
        results[c["name"]] = m
        # baseline delta on counts
        b = base.get(c["name"], {})
        d = ""
        for k in ("beams", "columns"):
            if k in b and b[k] != m[k]:
                d += f" {k}:{b[k]}→{m[k]}"
        flags = ""
        if m["floating"]:  flags += f" ⚠{m['floating']}float"
        if m["duplicate"]: flags += f" ⚠{m['duplicate']}dup"
        if m["overshoot"]: flags += f" ⚠{m['overshoot']}oversht"
        g = gt.get(c["name"], {})
        rec = ""
        if "beams" in g and g["beams"]:
            rec = f"  recall={m['beams']}/{g['beams']}={round(100*min(m['beams'],g['beams'])/g['beams'])}%"
        print(f"{c['name']:16} {m['beams']:>6} {m['joists']:>6} {m['columns']:>5} "
              f"{m['floating']:>6} {m['duplicate']:>4} {dt:>5}  "
              f"{(d or 'same') }{flags}{rec}")

    if save:
        json.dump(results, open(BASELINE, "w"), indent=2)
        print(f"\n[baseline saved → {os.path.basename(BASELINE)}]")
    else:
        print(f"\n(run with --save to set this as the regression baseline)")


if __name__ == "__main__":
    main_run()
