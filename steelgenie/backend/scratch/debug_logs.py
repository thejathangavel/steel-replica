import json
import os

brain_dir = r'C:\Users\user\.gemini\antigravity\brain'

for conv_id in os.listdir(brain_dir):
    log_path = os.path.join(brain_dir, conv_id, '.system_generated', 'logs', 'overview.txt')
    if not os.path.exists(log_path):
        continue
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if data.get('type') == 'PLANNER_RESPONSE':
                    for call in data.get('tool_calls', []):
                        args = call.get('args', {})
                        target = args.get('TargetFile', '')
                        if 'main.py' in target:
                            code = args.get('CodeContent', '') or args.get('ReplacementContent', '')
                            print(f"Conv: {conv_id}, Line: {i}, Tool: {call['name']}, Len: {len(code)}")
            except:
                continue
