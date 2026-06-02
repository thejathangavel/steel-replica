import requests

fname = '2025-07-10 - CB Milesburg - CCD 2 - Structural 3 - with dimensions.pdf'
r = requests.post('http://localhost:8000/analyse',
    json={'filename': fname, 'page_index': 0, 'scale_ratio': 96},
    timeout=120)
d = r.json()
members = d.get('members', [])
beams = [m for m in members if m['type'] == 'beam']
no_span = [b for b in beams if b.get('bx1') is None]
print(f'No-span beams: {len(no_span)}')
for b in sorted(no_span, key=lambda b: b['y']):
    prof = b['profile']
    bdir = b.get('beam_dir', '?')
    bx = b['x']
    by = b['y']
    print(f'  {prof:14s}  dir={bdir:3s}  x={bx:.3f}  y={by:.3f}')
