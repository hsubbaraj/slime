
import json
import argparse
import os
import math
from transformers import AutoTokenizer

# Constants
COST_PER_GPU_SEC = 0.001097
MODEL_PARAMS = 4e9  # Qwen3-4B
NUM_EPOCHS = 3
BASE_TFLOPS_PER_GPU = 350.0  # Conservative H100 utilization

# Heuristic from Audit Report:
# CP=2 (2 GPUs) fits ~24k tokens total.
# Implies ~12k tokens capacity per GPU after static overhead (weights/optim).
TOKENS_CAPACITY_PER_GPU = 12000 

def get_data_lengths(data_path, tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    lengths = []
    print(f"Analyzing data at {data_path}...")
    try:
        with open(data_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                messages = data['messages']
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                tokens = tokenizer.encode(text)
                lengths.append(len(tokens))
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []
        
    return sorted(lengths)

def estimate_metrics(num_gpus, cp_size, lengths):
    # Constraint 1: Max Sequence Length
    # With Ring Attention/CP, we can split the sequence across GPUs.
    # Capacity increases linearly with CP size.
    max_seq_len = cp_size * TOKENS_CAPACITY_PER_GPU
    
    # Filter Data
    valid_lengths = [l for l in lengths if l <= max_seq_len]
    valid_count = len(valid_lengths)
    total_count = len(lengths)
    valid_tokens = sum(valid_lengths)
    
    coverage = (valid_count / total_count) * 100 if total_count > 0 else 0
    
    if valid_count == 0:
        return None

    # Constraint 2: Throughput
    # CP introduces communication overhead.
    # Heuristic penalty: 5% per doubling of CP size.
    # CP1 = 1.0, CP2 = 0.95, CP4 = 0.90, CP8 = 0.85
    if cp_size == 1:
        cp_efficiency = 1.0
    else:
        cp_efficiency = 1.0 - (0.05 * math.log2(cp_size))
        
    # Throughput formula:
    # Tokens/sec = (Total TFLOPS * Efficiency) / (6 * Params)
    # Total TFLOPS = Num GPUs * Base TFLOPS
    
    total_tflops = num_gpus * BASE_TFLOPS_PER_GPU * 1e12
    effective_tflops = total_tflops * cp_efficiency
    
    throughput_tokens_per_sec = effective_tflops / (6 * MODEL_PARAMS)
    
    # Training Time
    # Total Tokens to train = valid_tokens * NUM_EPOCHS
    total_training_tokens = valid_tokens * NUM_EPOCHS
    training_time_sec = total_training_tokens / throughput_tokens_per_sec
    
    # Cost
    total_cost = training_time_sec * num_gpus * COST_PER_GPU_SEC
    
    return {
        "max_len": max_seq_len,
        "coverage": coverage,
        "valid_samples": valid_count,
        "time_min": training_time_sec / 60,
        "cost": total_cost,
        "throughput": throughput_tokens_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="Optimize SFT Cost")
    parser.add_argument("--data-path", default="slime/examples/tau-bench/data/airline_sft_formatted.jsonl")
    parser.add_argument("--tokenizer-path", default="models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    lengths = get_data_lengths(args.data_path, args.tokenizer_path)
    if not lengths:
        return

    print(f"\nTotal Conversations: {len(lengths)}")
    print(f"Max Length: {max(lengths)}")
    print("-" * 100)
    print(f"{'GPUs':<5} | {'CP':<3} | {'DP':<3} | {'MaxLen':<8} | {'Cover%':<6} | {'Time(m)':<8} | {'Cost($)':<8} | {'Notes'}")
    print("-" * 100)

    configs = []
    
    # Grid Search Options
    gpu_options = [1, 2, 4, 8]
    
    for num_gpus in gpu_options:
        # CP Size must divide Num GPUs (assuming we fill nodes/gpus fully)
        # And CP Size cannot be arbitrary, usually powers of 2 for Ring Attn
        possible_cp_sizes = [1, 2, 4, 8]
        
        for cp in possible_cp_sizes:
            if cp > num_gpus: continue
            if num_gpus % cp != 0: continue # Basic DP logic
            
            dp = num_gpus // cp
            
            metrics = estimate_metrics(num_gpus, cp, lengths)
            if metrics:
                metrics.update({"gpus": num_gpus, "cp": cp, "dp": dp})
                configs.append(metrics)

    # Sort by Cost
    configs.sort(key=lambda x: x['cost'])

    for c in configs:
        note = ""
        if c['coverage'] < 90:
            note = "Low Data!"
        elif c['cost'] == min(x['cost'] for x in configs if x['coverage'] > 95):
            note = "BEST VALUE (>95%)"
        elif c['coverage'] == 100 and c['cost'] == min(x['cost'] for x in configs if x['coverage'] == 100):
             note = "BEST FULL DATA"

        print(f"{c['gpus']:<5} | {c['cp']:<3} | {c['dp']:<3} | {c['max_len']:<8} | {c['coverage']:<6.1f} | {c['time_min']:<8.1f} | ${c['cost']:<7.2f} | {note}")

    print("-" * 100)
    print("\nAssumptions:")
    print(f"1. H100 Cost: ${COST_PER_GPU_SEC}/sec")
    print(f"2. Base H100 Util: {BASE_TFLOPS_PER_GPU} TFLOPS (Conservative)")
    print(f"3. Model: {MODEL_PARAMS/1e9:.1f}B Params")
    print(f"4. Epochs: {NUM_EPOCHS}")
    print(f"5. Memory Capacity: ~{TOKENS_CAPACITY_PER_GPU} tokens per GPU (Context Parallel)")
    print("6. CP Efficiency: 100% (CP1), 95% (CP2), 90% (CP4), 85% (CP8)")

if __name__ == "__main__":
    main()
