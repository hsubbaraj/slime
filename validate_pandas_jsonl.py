import pandas as pd
import sys

file_path = "slime/examples/tau-bench/data/airline_sft_formatted.jsonl"

try:
    print(f"Attempting to read {file_path} with pandas...")
    df = pd.read_json(file_path, lines=True, dtype={"label": str})
    print("Success! DataFrame loaded.")
    print(df.head())
except Exception as e:
    print(f"Pandas failed: {e}")
    sys.exit(1)
