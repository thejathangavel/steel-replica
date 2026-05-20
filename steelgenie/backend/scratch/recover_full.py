import json
import os

log_path = r'C:\Users\user\.gemini\antigravity\brain\1ad457de-b8d1-4d59-9278-eb1aecb33db1\.system_generated\logs\overview.txt'

best_code = ""
max_len = 0

if not os.path.exists(log_path):
    print(f"Log not found at {log_path}")
    exit(1)

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # The line might be very long, but json.loads should handle it
            data = json.loads(line)
            if data.get('type') == 'PLANNER_RESPONSE':
                for call in data.get('tool_calls', []):
                    if call.get('name') == 'write_to_file':
                        args = call.get('args')
                        target = args.get('TargetFile', '')
                        if 'main.py' in target:
                            code = args.get('CodeContent', '')
                            if len(code) > max_len:
                                max_len = len(code)
                                best_code = code
        except Exception as e:
            continue

if best_code:
    # Unescape
    # JSON strings in the log are already escaped. json.loads() handles one level.
    # But sometimes they are double escaped in the logs.
    if best_code.startswith('"') and best_code.endswith('"'):
        best_code = best_code[1:-1]
    
    # Standard JSON escapes
    processed_code = best_code.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\').replace('\\t', '\t')
    
    with open('recovered_main_full.py', 'w', encoding='utf-8') as out:
        out.write(processed_code)
    print(f"Recovered {len(processed_code)} bytes to recovered_main_full.py")
else:
    print("No code found in log")
