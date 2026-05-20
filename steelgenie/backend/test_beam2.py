import requests

r2 = requests.post('http://localhost:8000/analyse',
    json={'filename': 'Structural snaps.pdf', 'page_index': 0, 'scale_ratio': 96})
data = r2.json()
beams = [m for m in data['members'] if m['type'] == 'beam']
cols  = [m for m in data['members'] if m['type'] == 'column']

print(f"Columns: {len(cols)}   Beams: {len(beams)}")
print()

has_line = [b for b in beams if b.get('bx1') is not None]
no_line  = [b for b in beams if b.get('bx1') is None]
print(f"Beams with drawn-line endpoint : {len(has_line)}/{len(beams)}")
print(f"Beams using grid fallback      : {len(no_line)}/{len(beams)}")
print()
print("Top beams by span (vector-line matched):")
for b in sorted(has_line, key=lambda x: -x['length_ft'])[:12]:
    prof = b['profile']
    bdir = b['beam_dir']
    lft  = b['length_ft']
    x1   = b['bx1']
    y1   = b['by1']
    x2   = b['bx2']
    y2   = b['by2']
    print(f"  {prof:14s}  {bdir}  {lft:5.1f}ft  ({x1:.3f},{y1:.3f})->({x2:.3f},{y2:.3f})")
