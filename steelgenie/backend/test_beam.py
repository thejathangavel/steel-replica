import requests

# Upload
with open(r'C:\Users\user\Downloads\Structural snaps.pdf', 'rb') as f:
    r = requests.post('http://localhost:8000/upload',
                      files={'file': ('Structural snaps.pdf', f, 'application/pdf')})
print('Upload:', r.status_code, r.json().get('page_count'), 'pages')

# Analyse page 1 with 1/8" = 1'-0" scale (ratio=96)
r2 = requests.post('http://localhost:8000/analyse',
                   json={'filename': 'Structural snaps.pdf', 'page_index': 0, 'scale_ratio': 96})
data = r2.json()
members = data['members']
beams = [m for m in members if m['type'] == 'beam']
cols  = [m for m in members if m['type'] == 'column']

print(f'Total: {len(members)} members  —  {len(cols)} columns,  {len(beams)} beams')
print()
print('Beam samples (profile, dir, length_ft):')
for b in sorted(beams, key=lambda x: x['length_ft'], reverse=True)[:20]:
    prof = b['profile']
    bdir = b.get('beam_dir') or '?'
    blen = b['length_ft']
    print(f'  {prof:16s}  dir={bdir}  len={blen:5.1f} ft')

print()
print(f'Beams with length > 0 : {sum(1 for b in beams if b["length_ft"] > 0)}/{len(beams)}')
print(f'H beams               : {sum(1 for b in beams if b.get("beam_dir") == "H")}')
print(f'V beams               : {sum(1 for b in beams if b.get("beam_dir") == "V")}')
print(f'Elapsed               : {data.get("elapsed_seconds", data.get("elapsed"))}s')
