import requests
import json

url = "http://localhost:8000/analyse"
payload = {"filename": "NCU SherMan_Structural.pdf"}
headers = {"Content-Type": "application/json"}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=600)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
        data = response.json()
        print(f"Method: {data['extraction_method']}")
        print(f"Found {len(data['members'])} members.")
        print(f"Columns: {data['summary']['column']}")
        print(f"Nodes Found: {data.get('vertical_lines_found', 0)}") # actually this key is for vertical lines
        print(f"Debug Overlay: {data['debug_overlay']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
