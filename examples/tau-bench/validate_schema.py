
import json
import re
import os
import glob
from typing import Dict, Any, List
from tau_bench.envs.airline.tools import ALL_TOOLS

# Map tool names to their classes/schemas
TOOL_MAP = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}

def validate_tool_call(tool_name: str, args: Dict[str, Any]) -> bool:
    if tool_name not in TOOL_MAP:
        print(f"Error: Unknown tool '{tool_name}'")
        return False
    
    tool_class = TOOL_MAP[tool_name]
    # The tool class usually has an 'invoke' method or pydantic model for args
    # We can inspect the type hints of the 'invoke' method or the pydantic model if available.
    # In tau-bench, tools are typically Pydantic models or have a structure defining args.
    # Let's check against the 'get_info' schema which is standard OpenAI format.
    
    schema = tool_class.get_info()["function"]["parameters"]
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # Check required arguments
    for req in required:
        if req not in args:
            print(f"Error: Missing required argument '{req}' for tool '{tool_name}'")
            return False
            
    # Check unknown arguments
    for arg in args:
        if arg not in properties:
            print(f"Error: Unknown argument '{arg}' for tool '{tool_name}'")
            return False
            
    return True

def validate_file(filepath: str):
    print(f"Validating {filepath}...")
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    total_calls = 0
    valid_calls = 0
    
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            trajectory = ""
            if "metadata" in data and isinstance(data["metadata"], dict):
                trajectory = data["metadata"].get("trajectory", "")
            
            if not trajectory:
                trajectory = data.get("trajectory", "")

            # Extract function calls using regex
            # Pattern: <function>{"name": "...", "arguments": {...}}</function>
            # Handling potential multiline or spacing variations
            func_pattern = re.compile(r'<function>(.*?)</function>', re.DOTALL)
            matches = func_pattern.findall(trajectory)
            
            for match in matches:
                total_calls += 1
                try:
                    tool_call = json.loads(match)
                    name = tool_call.get("name")
                    args = tool_call.get("arguments", {})
                    
                    if validate_tool_call(name, args):
                        valid_calls += 1
                    else:
                        print(f"Failed validation at line {i+1}, tool: {name}")
                        
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in function block at line {i+1}: {match}")
                    
        except json.JSONDecodeError:
            print(f"Error: Invalid JSONL line {i+1}")
            
    print(f"File {filepath}: {valid_calls}/{total_calls} valid tool calls.")
    return valid_calls, total_calls

def main():
    data_dir = "slime/examples/tau-bench/data"
    files = glob.glob(os.path.join(data_dir, "tau_airline_mt_dialogs_*.jsonl"))
    
    if not files:
        print("No data files found.")
        return

    total_valid = 0
    total_total = 0
    
    for f in files:
        v, t = validate_file(f)
        total_valid += v
        total_total += t
        
    if total_total > 0:
        print(f"\nOverall Validation: {total_valid}/{total_total} ({total_valid/total_total*100:.2f}%)")
    else:
        print("\nOverall Validation: 0/0 (No tool calls found)")

if __name__ == "__main__":
    main()
