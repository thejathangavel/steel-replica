import json
import os

brain_dir = r'C:\Users\user\.gemini\antigravity\brain'
best_code = ""
max_len = 0

for conv_id in os.listdir(brain_dir):
    log_path = os.path.join(brain_dir, conv_id, '.system_generated', 'logs', 'overview.txt')
    if not os.path.exists(log_path):
        continue
    
    print(f"Checking {conv_id}...")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Check for tool calls in PLANNER_RESPONSE
                if data.get('type') == 'PLANNER_RESPONSE':
                    for call in data.get('tool_calls', []):
                        if call.get('name') in ['write_to_file', 'replace_file_content']:
                            args = call.get('args')
                            target = args.get('TargetFile', '')
                            if 'main.py' in target:
                                code = args.get('CodeContent', '') or args.get('ReplacementContent', '')
                                if len(code) > max_len:
                                    max_len = len(code)
                                    best_code = code
            except:
                continue

if best_code:
    if best_code.startswith('"') and best_code.endswith('"'):
        best_code = best_code[1:-1]
    
    processed_code = best_code.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\').replace('\\t', '\t')
    
    with open('recovered_main_ultimate.py', 'w', encoding='utf-8') as out:
        out.write(processed_code)
    print(f"Recovered {len(processed_code)} bytes to recovered_main_ultimate.py")
else:
    print("No code found in any log")
