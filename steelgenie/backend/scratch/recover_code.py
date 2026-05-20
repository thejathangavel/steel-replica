import json
import re

log_path = r'C:\Users\user\.gemini\antigravity\brain\1ad457de-b8d1-4d59-9278-eb1aecb33db1\.system_generated\logs\overview.txt'

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'PLANNER_RESPONSE':
                for call in data.get('tool_calls', []):
                    if call.get('name') == 'write_to_file':
                        args = call.get('args')
                        if args.get('TargetFile') == 'd:\\Steel-ghost\\steelgenie\\backend\\main.py' or args.get('TargetFile') == '"d:\\\\Steel-ghost\\\\steelgenie\\\\backend\\\\main.py"':
                            code = args.get('CodeContent')
                            # The code might be doubly escaped or quoted
                            if code.startswith('"') and code.endswith('"'):
                                code = code[1:-1]
                            # Replace escaped newlines
                            code = code.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            with open('recovered_main.py', 'w', encoding='utf-8') as out:
                                out.write(code)
                            print("Recovered to recovered_main.py")
        except:
            continue
