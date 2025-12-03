
import json
import re
import os
import glob
from typing import Dict, Any, List
from tau_bench.envs.airline.tools import ALL_TOOLS

# Map tool names to their classes/schemas
TOOL_MAP = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}

def is_valid_tool(tool_name: str) -> bool:
    return tool_name in TOOL_MAP

def parse_trajectory(traj_str: str) -> List[Dict[str, Any]]:
    # Regex to split the log into blocks
    # Blocks start with [TAG]: or [TAG - ...]:
    # We want to capture the tag and the content
    
    # Simplified regex to match [TAG]: content
    pattern = re.compile(r'\[(.*?)\]:\s*(.*?)(?=\n\[.*?\]:|$)', re.DOTALL)
    
    matches = pattern.findall(traj_str)
    events = []
    
    for tag, content in matches:
        tag = tag.strip()
        content = content.strip()
        
        if tag == "USER":
            events.append({"role": "user", "content": content})
        elif tag == "USER_THINKING":
            # Ignore user thinking for SFT
            pass
        elif tag == "ASSISTANT":
            # Check if it's a function call
            if "<function>" in content:
                # Extract function json
                func_match = re.search(r'<function>(.*?)</function>', content, re.DOTALL)
                if func_match:
                    try:
                        tool_data = json.loads(func_match.group(1))
                        events.append({"role": "assistant", "tool_calls": [
                            {"type": "function", "function": tool_data}
                        ]})
                    except:
                        # Failed to parse function, treat as text? Or discard?
                        # For now, treat as text if parse fails, but likely we should discard the conversation
                        events.append({"role": "assistant", "content": content})
            else:
                # Text response
                events.append({"role": "assistant", "content": content})
                
        elif tag == "ASSISTANT_THINKING":
            # We want to attach this to the *preceding* assistant message if possible,
            # or make it a new assistant message with <think> tags.
            # Strategy: Store as a separate event, merge later.
            events.append({"role": "assistant_thinking", "content": content})
            
        elif tag.startswith("FUNCTION_RESULT"):
            # Format: FUNCTION_RESULT - tool_name
            # content is the result
            tool_name = tag.split("-")[1].strip()
            events.append({"role": "tool", "name": tool_name, "content": content})
            
    return events

def merge_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = []
    
    for i, event in enumerate(events):
        role = event["role"]
        
        if role == "assistant_thinking":
            # Heuristic: If this follows a tool call (assistant message with tool_calls),
            # we typically want the thought *before* the tool call in the content field.
            # But here it comes *after* in the log.
            # Let's verify the previous message.
            if messages and messages[-1]["role"] == "assistant" and "tool_calls" in messages[-1]:
                # Merge into previous message content
                # Format: <think>...</think>
                # But wait, Qwen/OpenAI usually expects content (thought) THEN tool_calls.
                # If we add content now, it's fine.
                
                # Check if previous message already has content
                prev_content = messages[-1].get("content", "")
                new_content = f"<think>{event['content']}</think>"
                messages[-1]["content"] = prev_content + "\n" + new_content if prev_content else new_content
            
            elif messages and messages[-1]["role"] == "assistant":
                # Previous was text response. Append thought?
                # Usually thoughts come before response.
                # If the log has Response -> Think, it's post-hoc.
                # Let's prepend it?
                # Actually, in the log samples:
                # [ASSISTANT]: <function>...
                # [ASSISTANT_THINKING]: ...
                # It seems to be reasoning about the action just taken.
                
                # Let's stick to: put it in the content of the assistant message.
                prev_content = messages[-1].get("content", "")
                # Prepend or Append?
                # If we want the model to generate this, it should probably be Prepend if it's "planning",
                # but if it's "reflection", it might be Append?
                # Given Qwen reasoning models, <think> usually comes first.
                # Let's PREPEND it to the tool call message.
                new_content = f"<think>{event['content']}</think>"
                if not prev_content:
                    messages[-1]["content"] = new_content
                else:
                    # If there was already text, put think before it?
                    messages[-1]["content"] = new_content + "\n" + prev_content
            else:
                # Orphaned thought or start of turn?
                messages.append({"role": "assistant", "content": f"<think>{event['content']}</think>"})
                
        else:
            messages.append(event)
            
    return messages

def process_file(filepath: str, output_file):
    print(f"Processing {filepath}...")
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    valid_count = 0
    skipped_count = 0
    
    for line in lines:
        try:
            data = json.loads(line)
            trajectory = ""
            if "metadata" in data and isinstance(data["metadata"], dict):
                trajectory = data["metadata"].get("trajectory", "")
            if not trajectory:
                trajectory = data.get("trajectory", "")
                
            if not trajectory:
                continue
                
            events = parse_trajectory(trajectory)
            messages = merge_events(events)
            
            # Validate tools in messages
            is_valid = True
            invalid_tools_found = set()
            for msg in messages:
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        name = tc["function"]["name"]
                        if not is_valid_tool(name):
                            is_valid = False
                            invalid_tools_found.add(name)
                            break
                if not is_valid:
                    break
            
            if is_valid and messages:
                # Write to JSONL
                json.dump({"messages": messages}, output_file)
                output_file.write('\n')
                valid_count += 1
            else:
                skipped_count += 1
                if invalid_tools_found:
                    global ALL_INVALID_TOOLS
                    ALL_INVALID_TOOLS.update(invalid_tools_found)
                
        except json.JSONDecodeError:
            pass
            
    print(f"File {filepath}: {valid_count} valid, {skipped_count} skipped.")
    return valid_count, skipped_count

ALL_INVALID_TOOLS = set()

def main():
    data_dir = "slime/examples/tau-bench/data"
    output_path = os.path.join(data_dir, "airline_sft_formatted.jsonl")
    files = glob.glob(os.path.join(data_dir, "tau_airline_mt_dialogs_*.jsonl"))
    
    if not files:
        print("No data files found.")
        return

    total_valid = 0
    total_skipped = 0
    
    with open(output_path, 'w') as out_f:
        for f in files:
            v, s = process_file(f, out_f)
            total_valid += v
            total_skipped += s
        
    print(f"\nOverall: {total_valid} conversations saved to {output_path}. {total_skipped} skipped.")
    print(f"Invalid tools encountered: {ALL_INVALID_TOOLS}")

if __name__ == "__main__":
    main()
