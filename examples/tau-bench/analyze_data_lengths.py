"""Analyze token lengths in SFT data to understand filtering impact."""

import json
from transformers import AutoTokenizer
from collections import Counter

def analyze_sft_data(data_path, tokenizer_path, max_length=8192):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    lengths = []
    too_long = []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            messages = data['messages']

            # Apply chat template to get actual tokens
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(text)
            length = len(tokens)

            lengths.append(length)
            if length > max_length:
                too_long.append((i, length))

    # Statistics
    lengths_sorted = sorted(lengths)
    print(f"Total conversations: {len(lengths)}")
    print(f"\nLength statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"  Median: {lengths_sorted[len(lengths_sorted)//2]}")
    print(f"  P75: {lengths_sorted[int(len(lengths_sorted)*0.75)]}")
    print(f"  P90: {lengths_sorted[int(len(lengths_sorted)*0.90)]}")
    print(f"  P95: {lengths_sorted[int(len(lengths_sorted)*0.95)]}")
    print(f"  P99: {lengths_sorted[int(len(lengths_sorted)*0.99)]}")

    print(f"\n> {max_length} tokens: {len(too_long)} conversations ({len(too_long)/len(lengths)*100:.1f}%)")

    if too_long:
        print(f"\nConversations exceeding {max_length} tokens:")
        for idx, length in too_long[:10]:  # Show first 10
            print(f"  Line {idx}: {length} tokens")
        if len(too_long) > 10:
            print(f"  ... and {len(too_long)-10} more")

    # Bucket analysis
    buckets = [0, 2048, 4096, 6144, 8192, 10240, 12288, 16384, max(lengths)+1]
    bucket_counts = Counter()
    for length in lengths:
        for i in range(len(buckets)-1):
            if buckets[i] <= length < buckets[i+1]:
                bucket_counts[f"{buckets[i]}-{buckets[i+1]}"] += 1
                break

    print(f"\nToken length distribution:")
    for bucket in sorted(bucket_counts.keys(), key=lambda x: int(x.split('-')[0])):
        count = bucket_counts[bucket]
        pct = count / len(lengths) * 100
        print(f"  {bucket}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    analyze_sft_data(
        "examples/tau-bench/data/airline_sft_formatted.jsonl",
        "/root/models/Qwen3-4B-Instruct-2507",  # Change locally
        max_length=8192
    )
