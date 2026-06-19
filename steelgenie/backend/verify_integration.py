"""
Brace Integration Verification Script
---------------------------------------
Calls /analyse with detect_braces=false (baseline) and detect_braces=true
across multiple PDFs and pages.  Collects:
  - timing (baseline vs brace-enabled)
  - member counts (beams, braces)
  - brace field completeness
  - duplicate detection
  - sample API response excerpts

Outputs: verify_results.json
"""
import json, time, math, sys, io, requests
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

# (label, filename, page_index, scale_ratio)
TEST_CASES = [
    ("snaps_elevation",   "Structural snaps.pdf",            6, 96.0),
    ("snaps_elevation2",  "Structural snaps.pdf",            7, 96.0),
    ("snaps_framing",     "Structural snaps.pdf",            2, 96.0),
    ("07comb_elevation",  "07_STRUCTURAL_COMBINED.pdf",      47, 192.0),
    ("07comb_elevation2", "07_STRUCTURAL_COMBINED.pdf",      48, 192.0),
    ("binder_framing",    "#Structural binder.pdf",          12, 64.0),
    ("bayhealth_framing", "2026.03.27_Bayhealth Sussex MOB_DD Set_Structural.pdf", 5, 192.0),
    ("spruce_framing",    "02 Struct 98 Spruce_2026-03-13_BID.pdf", 3, 96.0),
]

REQUIRED_BRACE_FIELDS = {"type", "confidence", "length_ft", "angle_deg",
                          "context", "bx1", "by1", "bx2", "by2", "x", "y"}


def call_analyse(filename, page_index, scale_ratio, detect_braces):
    payload = {
        "filename":      filename,
        "page_index":    page_index,
        "scale_ratio":   scale_ratio,
        "detect_braces": detect_braces,
    }
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/analyse", json=payload, timeout=120)
    elapsed = round(time.time() - t0, 2)
    r.raise_for_status()
    return r.json(), elapsed


def check_brace_fields(member):
    missing = []
    for f in REQUIRED_BRACE_FIELDS:
        if member.get(f) is None and f not in ("profile", "label"):
            # confidence / context / length_ft / angle_deg must be non-None
            if f in ("confidence", "length_ft", "angle_deg", "context"):
                missing.append(f)
            # coordinate fields must be numeric
            elif f in ("bx1","by1","bx2","by2","x","y"):
                if not isinstance(member.get(f), (int, float)):
                    missing.append(f)
    return missing


def detect_duplicates(members):
    """
    Find brace members whose bx1/by1/bx2/by2 differ by < 0.005 fractional units
    (≈ 0.5% of page dimension — effectively the same drawn segment).
    """
    braces = [m for m in members if m.get("type") == "brace"]
    dupes = []
    for i in range(len(braces)):
        for j in range(i + 1, len(braces)):
            a, b = braces[i], braces[j]
            d = math.hypot(a["bx1"]-b["bx1"], a["by1"]-b["by1"]) + \
                math.hypot(a["bx2"]-b["bx2"], a["by2"]-b["by2"])
            # Also check reversed direction
            dr = math.hypot(a["bx1"]-b["bx2"], a["by1"]-b["by2"]) + \
                 math.hypot(a["bx2"]-b["bx1"], a["by2"]-b["by1"])
            if min(d, dr) < 0.01:
                dupes.append((i, j))
    return dupes


print("=" * 70)
print("  BRACE INTEGRATION VERIFICATION")
print("=" * 70)

all_results = []

baseline_times = []
brace_times    = []

for (label, filename, page_index, scale_ratio) in TEST_CASES:
    print(f"\n  [{label}]  page={page_index}  scale=1:{scale_ratio:.0f}")

    # ── Baseline (no braces) ─────────────────────────────────────────────────
    try:
        base_resp, base_time = call_analyse(filename, page_index, scale_ratio, False)
        baseline_times.append(base_time)
        base_beams  = [m for m in base_resp["members"] if m.get("type") != "brace"]
        base_braces = [m for m in base_resp["members"] if m.get("type") == "brace"]
        print(f"    Baseline  : {base_time:.2f}s  beams={len(base_beams)}  "
              f"braces={len(base_braces)}  total={len(base_resp['members'])}")
    except Exception as e:
        print(f"    Baseline  : ERROR — {e}")
        base_resp, base_time = None, 0
        base_beams = []

    # ── Brace-enabled ────────────────────────────────────────────────────────
    try:
        brace_resp, brace_time = call_analyse(filename, page_index, scale_ratio, True)
        brace_times.append(brace_time)
        all_beams  = [m for m in brace_resp["members"] if m.get("type") != "brace"]
        all_braces = [m for m in brace_resp["members"] if m.get("type") == "brace"]
        high_b     = [b for b in all_braces if b.get("confidence") == "HIGH"]
        med_b      = [b for b in all_braces if b.get("confidence") == "MEDIUM"]
        print(f"    Braces ON : {brace_time:.2f}s  beams={len(all_beams)}  "
              f"braces={len(all_braces)} (HIGH={len(high_b)} MED={len(med_b)})  "
              f"total={len(brace_resp['members'])}")
    except Exception as e:
        print(f"    Braces ON : ERROR — {e}")
        brace_resp, brace_time = None, 0
        all_beams = []
        all_braces = []
        high_b = med_b = []

    # ── Beam count stability check ───────────────────────────────────────────
    beam_stable = len(all_beams) == len(base_beams)
    print(f"    Beam count stable: {'YES' if beam_stable else 'NO — REGRESSION'} "
          f"({len(base_beams)} → {len(all_beams)})")

    # ── Field completeness ───────────────────────────────────────────────────
    field_errors = 0
    for b in all_braces:
        missing = check_brace_fields(b)
        if missing:
            print(f"      [FIELD ERROR] brace missing: {missing}")
            field_errors += 1
    if all_braces and field_errors == 0:
        print(f"    Field completeness: PASS ({len(all_braces)} braces, all fields present)")

    # ── Duplicate check ──────────────────────────────────────────────────────
    if brace_resp:
        dupes = detect_duplicates(brace_resp["members"])
        if dupes:
            print(f"    Duplicates: {len(dupes)} DETECTED — {dupes[:3]}")
        else:
            print(f"    Duplicates: NONE")

    # ── Context distribution ─────────────────────────────────────────────────
    ctx_counts = defaultdict(int)
    for b in all_braces:
        ctx_counts[b.get("context", "?")] += 1
    if ctx_counts:
        print(f"    Context: {dict(ctx_counts)}")

    # ── Overhead ─────────────────────────────────────────────────────────────
    overhead = round(brace_time - base_time, 2)
    pct      = round(100 * overhead / base_time, 1) if base_time > 0 else 0
    print(f"    Overhead: {overhead:+.2f}s  ({pct:+.1f}%)")

    all_results.append({
        "label":          label,
        "filename":       filename,
        "page_index":     page_index,
        "scale_ratio":    scale_ratio,
        "baseline_time":  base_time,
        "brace_time":     brace_time,
        "overhead_s":     overhead,
        "overhead_pct":   pct,
        "base_beams":     len(base_beams),
        "brace_beams":    len(all_beams),
        "beam_stable":    beam_stable,
        "braces_high":    len(high_b),
        "braces_medium":  len(med_b),
        "braces_total":   len(all_braces),
        "field_errors":   field_errors,
        "duplicates":     len(detect_duplicates(brace_resp["members"])) if brace_resp else -1,
        "context":        dict(ctx_counts),
        "sample_brace":   all_braces[0] if all_braces else None,
    })

# ── Global summary ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  GLOBAL SUMMARY")
print(f"{'='*70}")

total_braces = sum(r["braces_total"] for r in all_results)
total_high   = sum(r["braces_high"]  for r in all_results)
all_stable   = all(r["beam_stable"]  for r in all_results)
all_nodup    = all(r["duplicates"] == 0 for r in all_results if r["duplicates"] >= 0)
all_fields   = all(r["field_errors"] == 0 for r in all_results)

avg_base  = sum(baseline_times) / len(baseline_times) if baseline_times else 0
avg_brace = sum(brace_times)    / len(brace_times)    if brace_times    else 0

print(f"\n  Test cases        : {len(all_results)}")
print(f"  Total braces      : {total_braces} (HIGH={total_high})")
print(f"  Beam count stable : {'YES — all pages' if all_stable else 'NO — regressions detected'}")
print(f"  No duplicates     : {'YES' if all_nodup else 'NO — dupes found'}")
print(f"  Field completeness: {'PASS' if all_fields else 'FAIL'}")
print(f"\n  Avg time baseline : {avg_base:.2f}s")
print(f"  Avg time w/braces : {avg_brace:.2f}s")
print(f"  Avg overhead      : {avg_brace - avg_base:+.2f}s  "
      f"({100*(avg_brace-avg_base)/avg_base:.1f}%)" if avg_base > 0 else "")

# ── Save results ──────────────────────────────────────────────────────────────
out_path = "verify_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved: {out_path}")
print("=" * 70)
