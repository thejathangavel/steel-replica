import requests, time
from collections import Counter

# Image test
t0 = time.time()
r = requests.post("http://localhost:8000/analyse",
    json={"filename": "sample img.png", "page_index": 0, "scale_ratio": 96},
    timeout=180)
elapsed = round(time.time() - t0, 1)
data = r.json()
members = data.get("members", [])
beams = [m for m in members if m["type"] == "beam"]
cols  = [m for m in members if m["type"] == "column"]

print(f"=== IMAGE: {len(cols)} columns   {len(beams)} beams  ({elapsed}s) ===")
bc = Counter(b["profile"] for b in beams)
print("BEAMS:")
for prof, cnt in sorted(bc.items()):
    print(f"  {prof:14s} x{cnt}")
cc = Counter(c["profile"] for c in cols)
print("COLS:")
for prof, cnt in sorted(cc.items()):
    print(f"  {prof:14s} x{cnt}")
h_b = sum(1 for b in beams if b.get("beam_dir") == "H")
v_b = sum(1 for b in beams if b.get("beam_dir") == "V")
print(f"Dir: H={h_b}  V={v_b}")

print()
print("--- Vector PDF sanity check ---")
r2 = requests.post("http://localhost:8000/analyse",
    json={"filename": "Structural snaps.pdf", "page_index": 0, "scale_ratio": 96},
    timeout=60)
d2 = r2.json()
b2 = [m for m in d2.get("members", []) if m["type"] == "beam"]
c2 = [m for m in d2.get("members", []) if m["type"] == "column"]
el2 = d2.get("elapsed")
print(f"PDF: {len(c2)} columns  {len(b2)} beams  ({el2}s)")
bc2 = Counter(b["profile"] for b in b2)
print("PDF BEAMS:")
for prof, cnt in sorted(bc2.items()):
    print(f"  {prof:14s} x{cnt}")
