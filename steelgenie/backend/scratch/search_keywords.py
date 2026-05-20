import json
import os

brain_dir = r'C:\Users\user\.gemini\antigravity\brain'
keywords = ['schedule', 'weight', 'tons', 'formula', 'connection', 'AISI', 'AISC', 'filtered_profiles']

for conv_id in os.listdir(brain_dir):
    log_path = os.path.join(brain_dir, conv_id, '.system_generated', 'logs', 'overview.txt')
    if not os.path.exists(log_path):
        continue
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            try:
                if any(kw in line for kw in keywords):
                    # Try to see if it's a tool call with code
                    data = json.loads(line)
                    if data.get('type') == 'PLANNER_RESPONSE':
                        for call in data.get('tool_calls', []):
                            args = call.get('args', {})
                            code = args.get('CodeContent', '') or args.get('ReplacementContent', '')
                            if code:
                                print(f"MATCH in {conv_id} line {i}: {call['name']} (len {len(code)})")
                                # Print first 100 chars to identify
                                print(f"Preview: {code[:200]}...")
            except:
                continue
