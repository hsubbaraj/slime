import json
import sys

file_path = "slime/examples/tau-bench/data/airline_sft_formatted.jsonl"

try:
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue # Skip empty lines
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error on line {i+1}: {e}")
                print(f"Line content: {line[:100]}...")
                sys.exit(1)
    print("All lines are valid JSON.")

except FileNotFoundError:
    print(f"File not found: {file_path}")
