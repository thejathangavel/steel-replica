import requests

files = [
    ('#Structural binder.pdf', 96),
    ('2025-07-10 - CB Milesburg - CCD 2 - Structural 3 - with dimensions.pdf', 96),
    ('E2501_1.pdf', 96),
    ('E2502_1.pdf', 96),
    ('E2505_0.pdf', 96),
    ('E2701_A.pdf', 96),
    ('E3303_A.pdf', 96),
    ('E3705_A.pdf', 96),
    ('Structural snaps.pdf', 96),
    ('STUD COUNT 2.pdf', 96),
    ('Latest_Structural dwg_Binder (Addendum-02).pdf', 96),
]

for fname, ratio in files:
    try:
        r = requests.post('http://localhost:8000/analyse',
            json={'filename': fname, 'page_index': 0, 'scale_ratio': ratio},
            timeout=120)
        if r.status_code == 200:
            d = r.json()
            members = d.get('members', [])
            beams = [m for m in members if m['type'] == 'beam']
            no_span = [b for b in beams if b.get('bx1') is None]
            print(f'OK  : {fname[:55]:55s}  beams={len(beams):3d}  no_span={len(no_span)}')
        else:
            print(f'FAIL: {fname[:55]}  status={r.status_code}')
    except Exception as e:
        print(f'ERR : {fname[:55]}  {e}')
