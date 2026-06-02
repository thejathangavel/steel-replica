import requests

r = requests.post('http://localhost:8000/analyse',
    json={'filename': 'STUD COUNT 2.pdf', 'page_index': 0, 'scale_ratio': 96},
    timeout=120)
d = r.json()
members = d.get('members', [])
beams = [m for m in members if m['type'] == 'beam']
no_span = [b for b in beams if b.get('bx1') is None]
print(f'Total beams: {len(beams)}, no_span: {len(no_span)}')
for b in no_span:
    bdir = b.get('beam_dir') or '?'
    print(f'  {b["profile"]:14s}  dir={bdir}  x={b["x"]:.3f}  y={b["y"]:.3f}  len={b.get("length_ft")}')
