import os
import subprocess

import modal
from modal import App, Image, Volume
from modal.mount import Mount

# Define Volumes
data_volume = Volume.from_name("slime-data", create_if_missing=True)
model_volume = Volume.from_name("slime-models", create_if_missing=True)
checkpoint_volume = Volume.from_name("slime-checkpoints", create_if_missing=True)

# Path Constants within the Modal container
SLIME_REMOTE_PATH = "/root/slime"
TAU_BENCH_REMOTE_PATH = "/root/tau-bench"
MEGATRON_REMOTE_PATH = "/root/Megatron-LM"  # Expected in slime docker image
DATA_REMOTE_PATH = "/root/data"
MODEL_REMOTE_PATH = "/root/models"
CHECKPOINT_REMOTE_PATH = "/root/checkpoints"

# Local paths (Resolved relative to this script file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_LOCAL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../slime"))  # slime root
TAU_BENCH_LOCAL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../tau-bench"))  # tau-bench root
DATASET_LOCAL_PATH = os.path.join(SLIME_LOCAL_PATH, "slime/examples/tau-bench/data/airline_sft_formatted.jsonl")

print(f"SLIME_LOCAL_PATH: {SLIME_LOCAL_PATH}")
print(f"TAU_BENCH_LOCAL_PATH: {TAU_BENCH_LOCAL_PATH}")
print(f"DATASET_LOCAL_PATH: {DATASET_LOCAL_PATH}")

# Verify dataset path exists locally (optional debug)
if not os.path.exists(DATASET_LOCAL_PATH):
    # Try alternate path if structure is different
    DATASET_LOCAL_PATH = os.path.join(SCRIPT_DIR, "data", "airline_sft_formatted.jsonl")

# Define the image using slime's official Docker image
image = (
    Image.from_registry("slimerl/slime:latest")
    .add_local_dir(SLIME_LOCAL_PATH, remote_path=SLIME_REMOTE_PATH)
    .add_local_dir(TAU_BENCH_LOCAL_PATH, remote_path=TAU_BENCH_REMOTE_PATH)
)

app = App("slime-tau-sft", image=image)


@app.function(
    gpu="H100",  # Or A100-80GB, adjust as needed
    timeout=86400,  # 24 hours
    volumes={
        DATA_REMOTE_PATH: data_volume,
        MODEL_REMOTE_PATH: model_volume,
        CHECKPOINT_REMOTE_PATH: checkpoint_volume,
    },
)
def run_sft_job():
    # Install slime and tau-bench in editable mode within the container
    print("Installing slime in editable mode...")
    subprocess.run(["pip", "install", "-e", SLIME_REMOTE_PATH], check=True)
    print("Installing tau-bench in editable mode...")
    subprocess.run(["pip", "install", "-e", TAU_BENCH_REMOTE_PATH], check=True)

    # Ensure scripts are executable
    sft_script_path = os.path.join(SLIME_REMOTE_PATH, "examples", "tau-bench", "retool_sft.sh")
    subprocess.run(["chmod", "+x", sft_script_path], check=True)

    # Define environment variables for the training script
    env = os.environ.copy()
    env["SLIME_HOME"] = SLIME_REMOTE_PATH
    env["MODEL_PATH"] = os.path.join(MODEL_REMOTE_PATH, "Qwen3-4B-Instruct-2507")  # Qwen3-4B-Instruct-2507 model
    env["DATA_PATH"] = os.path.join(DATA_REMOTE_PATH, "airline_sft_formatted.jsonl")
    env["OUTPUT_PATH"] = os.path.join(CHECKPOINT_REMOTE_PATH, "Qwen3-4B-sft-airline")
    env["MEGATRON_PATH"] = MEGATRON_REMOTE_PATH
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Run the SFT script
    print("Launching SFT Training...")
    result = subprocess.run([sft_script_path], env=env, text=True, capture_output=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}")

    print("Training completed successfully.")


@app.function(
    gpu="H100:2",
    timeout=60 * 15,  # 15 minutes - allows for data loading + 1-2 training steps
    volumes={
        DATA_REMOTE_PATH: data_volume,
        MODEL_REMOTE_PATH: model_volume,
        CHECKPOINT_REMOTE_PATH: checkpoint_volume,
    },
)
def run_quick_test():
    """Quick test with just 1-2 sequences to verify setup."""
    print("Starting QUICK Test (1-2 sequences only)...")

    # Install slime and tau-bench in editable mode within the container
    subprocess.run(["pip", "install", "-e", SLIME_REMOTE_PATH], check=True)
    subprocess.run(["pip", "install", "-e", TAU_BENCH_REMOTE_PATH], check=True)

    # Ensure scripts are executable
    sft_script_path = os.path.join(SLIME_REMOTE_PATH, "examples", "tau-bench", "retool_sft.sh")
    subprocess.run(["chmod", "+x", sft_script_path], check=True)

    # Define environment variables for the training script
    env = os.environ.copy()
    env["SLIME_HOME"] = SLIME_REMOTE_PATH
    env["MODEL_PATH"] = os.path.join(MODEL_REMOTE_PATH, "Qwen3-4B-Instruct-2507")
    env["DATA_PATH"] = os.path.join(DATA_REMOTE_PATH, "airline_sft_tiny_test.jsonl")  # TINY dataset
    env["OUTPUT_PATH"] = os.path.join(CHECKPOINT_REMOTE_PATH, "Qwen3-4B-sft-quick-test")
    env["MEGATRON_PATH"] = MEGATRON_REMOTE_PATH
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Quick test overrides
    env["NUM_EPOCH"] = "1"
    env["BATCH_SIZE"] = "1"
    env["ACTOR_NUM_GPUS_PER_NODE"] = "2"
    env["GLOBAL_BATCH_SIZE"] = "1"

    # Run the SFT script
    print("Launching QUICK SFT Test (1-2 sequences)...")

    # Save full logs to checkpoint volume for debugging
    log_path = os.path.join(CHECKPOINT_REMOTE_PATH, "quick_test.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run([sft_script_path], env=env, text=True, stdout=log_f, stderr=subprocess.STDOUT)

    # Commit the volume to persist the log
    checkpoint_volume.commit()

    # Print last 100 lines for immediate feedback
    print("Last 100 lines of output:")
    with open(log_path, "r") as log_f:
        lines = log_f.readlines()
        print("".join(lines[-100:]))

    if result.returncode != 0:
        raise Exception(f"Quick Test failed with exit code {result.returncode}. Check {log_path} for full logs.")

    print("✓ Quick Test completed successfully!")
    print("  Processed 1-2 conversations")
    print("  Ready for full training run")


@app.function(
    gpu="H100:2",  # Request 2 H100 GPUs for the test job
    timeout=60 * 10,  # 10 minutes timeout
    volumes={
        DATA_REMOTE_PATH: data_volume,
        MODEL_REMOTE_PATH: model_volume,
        CHECKPOINT_REMOTE_PATH: checkpoint_volume,
    },
)
def run_test_job():
    print("Starting Test Job (H100, 1 Epoch)...")

    # Install slime and tau-bench in editable mode within the container
    subprocess.run(["pip", "install", "-e", SLIME_REMOTE_PATH], check=True)
    subprocess.run(["pip", "install", "-e", TAU_BENCH_REMOTE_PATH], check=True)

    # Ensure scripts are executable
    sft_script_path = os.path.join(SLIME_REMOTE_PATH, "examples", "tau-bench", "retool_sft.sh")
    subprocess.run(["chmod", "+x", sft_script_path], check=True)

    # Define environment variables for the training script
    env = os.environ.copy()
    env["SLIME_HOME"] = SLIME_REMOTE_PATH
    env["MODEL_PATH"] = os.path.join(MODEL_REMOTE_PATH, "Qwen3-4B-Instruct-2507")
    env["DATA_PATH"] = os.path.join(DATA_REMOTE_PATH, "airline_sft_filtered.jsonl")  # Use filtered data
    env["OUTPUT_PATH"] = os.path.join(CHECKPOINT_REMOTE_PATH, "Qwen3-4B-sft-airline-test")
    env["MEGATRON_PATH"] = MEGATRON_REMOTE_PATH
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Test specific overrides
    env["NUM_EPOCH"] = "1"
    env["BATCH_SIZE"] = "1"  # MUST be 1 when CP=2 on 2 GPUs
    # Configure for 2 GPUs
    env["ACTOR_NUM_GPUS_PER_NODE"] = "2"
    env["GLOBAL_BATCH_SIZE"] = "1"  # MUST be 1 when CP=2 on 2 GPUs

    # Run the SFT script
    print("Launching SFT Test Training...")

    # Save full logs to checkpoint volume for debugging
    log_path = os.path.join(CHECKPOINT_REMOTE_PATH, "train_full.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run([sft_script_path], env=env, text=True, stdout=log_f, stderr=subprocess.STDOUT)

    # Commit the volume to persist the log
    checkpoint_volume.commit()

    # Print last 100 lines for immediate feedback
    print("Last 100 lines of output:")
    with open(log_path, "r") as log_f:
        lines = log_f.readlines()
        print("".join(lines[-100:]))

    if result.returncode != 0:
        raise Exception(f"Test Training failed with exit code {result.returncode}. Check {log_path} for full logs.")

    print("Test Training completed successfully.")


@app.function(
    gpu="H100",
    timeout=3600,  # 1 hour should be plenty for conversion
    volumes={MODEL_REMOTE_PATH: model_volume},
)
def run_conversion_job():
    print("Starting Model Conversion Job...")

    # Install slime in editable mode (dependencies)
    subprocess.run(["pip", "install", "-e", SLIME_REMOTE_PATH], check=True)

    # Ensure script is executable
    script_path = os.path.join(SLIME_REMOTE_PATH, "examples", "tau-bench", "convert_model.sh")
    subprocess.run(["chmod", "+x", script_path], check=True)

    # Define env
    env = os.environ.copy()
    env["SLIME_HOME"] = SLIME_REMOTE_PATH
    env["MODEL_ROOT"] = MODEL_REMOTE_PATH
    env["MEGATRON_PATH"] = MEGATRON_REMOTE_PATH

    # Run conversion
    result = subprocess.run([script_path], env=env, text=True, capture_output=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise Exception(f"Conversion failed with exit code {result.returncode}")

    print("Model conversion successful.")


@app.function(
    timeout=300,
    volumes={DATA_REMOTE_PATH: data_volume},
)
def create_tiny_test_data(num_samples=2):
    """Create a tiny test dataset with just N conversations for quick testing."""
    import json

    input_path = os.path.join(DATA_REMOTE_PATH, "airline_sft_filtered.jsonl")
    output_path = os.path.join(DATA_REMOTE_PATH, "airline_sft_tiny_test.jsonl")

    print(f"Creating tiny test dataset with {num_samples} conversations...")

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            if i >= num_samples:
                break
            outfile.write(line)

    data_volume.commit()

    print(f"✓ Created test dataset: {output_path}")
    print(f"  Contains {num_samples} conversations")
    return {"samples": num_samples, "output_path": output_path}


@app.function(
    timeout=600,
    volumes={DATA_REMOTE_PATH: data_volume, MODEL_REMOTE_PATH: model_volume},
)
def filter_long_conversations(max_tokens=24576):  # 12K per GPU × 2 GPUs with CP=2
    """Filter conversations to remove those exceeding max_tokens."""
    import json

    from transformers import AutoTokenizer

    input_path = os.path.join(DATA_REMOTE_PATH, "airline_sft_formatted.jsonl")
    output_path = os.path.join(DATA_REMOTE_PATH, "airline_sft_filtered.jsonl")
    tokenizer_path = os.path.join(MODEL_REMOTE_PATH, "Qwen3-4B-Instruct-2507")

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Filtering conversations from {input_path}...")
    print(f"Max tokens: {max_tokens:,}")

    kept = 0
    filtered = 0

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            data = json.loads(line)
            messages = data["messages"]

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

    data_volume.commit()

    print(f"\n{'=' * 60}")
    print(f"FILTERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Kept: {kept} conversations")
    print(f"  Filtered: {filtered} conversations ({filtered / (kept + filtered) * 100:.1f}%)")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    return {"kept": kept, "filtered": filtered, "output_path": output_path}


@app.function(
    timeout=600,
    volumes={DATA_REMOTE_PATH: data_volume, MODEL_REMOTE_PATH: model_volume},
)
def analyze_data_lengths():
    """Analyze token lengths in SFT data."""
    import json
    from collections import Counter

    from transformers import AutoTokenizer

    data_path = os.path.join(DATA_REMOTE_PATH, "airline_sft_formatted.jsonl")
    tokenizer_path = os.path.join(MODEL_REMOTE_PATH, "Qwen3-4B-Instruct-2507")
    max_length = 8192

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Analyzing data from {data_path}...")
    lengths = []
    too_long = []

    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            messages = data["messages"]

            # Apply chat template to get actual tokens
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(text)
            length = len(tokens)

            lengths.append(length)
            if length > max_length:
                too_long.append((i, length))

    # Statistics
    lengths_sorted = sorted(lengths)
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS RESULTS")
    print(f"{'=' * 60}")
    print(f"Total conversations: {len(lengths)}")
    print(f"\nLength statistics:")
    print(f"  Min: {min(lengths):,} tokens")
    print(f"  Max: {max(lengths):,} tokens")
    print(f"  Mean: {sum(lengths) / len(lengths):,.1f} tokens")
    print(f"  Median: {lengths_sorted[len(lengths_sorted) // 2]:,} tokens")
    print(f"  P75: {lengths_sorted[int(len(lengths_sorted) * 0.75)]:,} tokens")
    print(f"  P90: {lengths_sorted[int(len(lengths_sorted) * 0.90)]:,} tokens")
    print(f"  P95: {lengths_sorted[int(len(lengths_sorted) * 0.95)]:,} tokens")
    print(f"  P99: {lengths_sorted[int(len(lengths_sorted) * 0.99)]:,} tokens")

    print(f"\n> {max_length:,} tokens: {len(too_long)} conversations ({len(too_long) / len(lengths) * 100:.1f}%)")

    if too_long:
        print(f"\nConversations exceeding {max_length:,} tokens:")
        for idx, length in too_long[:20]:  # Show all if < 20
            print(f"  Line {idx}: {length:,} tokens")
        if len(too_long) > 20:
            print(f"  ... and {len(too_long) - 20} more")

    # Bucket analysis
    buckets = [0, 2048, 4096, 6144, 8192, 10240, 12288, 16384, max(lengths) + 1]
    bucket_counts = Counter()
    for length in lengths:
        for i in range(len(buckets) - 1):
            if buckets[i] <= length < buckets[i + 1]:
                bucket_counts[f"{buckets[i]}-{buckets[i + 1]}"] += 1
                break

    print(f"\nToken length distribution:")
    for bucket_range in sorted(bucket_counts.keys(), key=lambda x: int(x.split("-")[0])):
        count = bucket_counts[bucket_range]
        pct = count / len(lengths) * 100
        bar = "█" * int(pct / 2)  # Simple bar chart
        print(f"  {bucket_range:>12}: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION:")
    if len(too_long) <= len(lengths) * 0.05:  # < 5%
        print(f"✓ Filtering to {max_length:,} tokens removes only {len(too_long) / len(lengths) * 100:.1f}% of data")
        print(f"  This is acceptable for most use cases.")
    else:
        print(f"⚠ Filtering to {max_length:,} tokens removes {len(too_long) / len(lengths) * 100:.1f}% of data")
        print(f"  Consider using Context Parallel (--context-parallel-size 2)")
    print(f"{'=' * 60}\n")
