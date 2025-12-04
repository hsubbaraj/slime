"""Filter out conversations that exceed max token length."""

import json
import sys
from transformers import AutoTokenizer

def filter_conversations(input_path, output_path, tokenizer_path, max_tokens=30720):
    """Filter conversations to keep only those under max_tokens."""

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Filtering conversations from {input_path}...")
    kept = 0
    filtered = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for i, line in enumerate(infile):
            data = json.loads(line)
            messages = data['messages']

            # Apply chat template to get actual tokens
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(text)
            length = len(tokens)

            if length <= max_tokens:
                outfile.write(line)
                kept += 1
            else:
                filtered += 1
                print(f"  Filtered line {i}: {length:,} tokens > {max_tokens:,}")

    print(f"\nResults:")
    print(f"  Kept: {kept} conversations")
    print(f"  Filtered: {filtered} conversations ({filtered/(kept+filtered)*100:.1f}%)")
    print(f"  Output: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python filter_long_conversations.py INPUT OUTPUT TOKENIZER_PATH [MAX_TOKENS]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    tokenizer_path = sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 30720

    filter_conversations(input_path, output_path, tokenizer_path, max_tokens)
