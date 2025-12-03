import modal
from modal import App, Image, Volume
from modal.mount import Mount
import os
import subprocess



# Define Volumes
data_volume = Volume.from_name("slime-data", create_if_missing=True)
model_volume = Volume.from_name("slime-models", create_if_missing=True)
checkpoint_volume = Volume.from_name("slime-checkpoints", create_if_missing=True)

# Path Constants within the Modal container
SLIME_REMOTE_PATH = "/root/slime"
TAU_BENCH_REMOTE_PATH = "/root/tau-bench"
MEGATRON_REMOTE_PATH = "/root/Megatron-LM" # Expected in slime docker image
DATA_REMOTE_PATH = "/root/data"
MODEL_REMOTE_PATH = "/root/models"
CHECKPOINT_REMOTE_PATH = "/root/checkpoints"

# Local paths (Resolved relative to this script file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_LOCAL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../slime")) # slime root
TAU_BENCH_LOCAL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../tau-bench")) # tau-bench root
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
    Image.from_registry("slimerl/slime:latest").
    add_local_dir(SLIME_LOCAL_PATH, remote_path=SLIME_REMOTE_PATH).
    add_local_dir(TAU_BENCH_LOCAL_PATH, remote_path=TAU_BENCH_REMOTE_PATH)
)

app = App("slime-tau-sft", image=image)

@app.function(
    gpu="H100",  # Or A100-80GB, adjust as needed
    timeout=86400, # 24 hours
    volumes={
        DATA_REMOTE_PATH: data_volume,
        MODEL_REMOTE_PATH: model_volume,
        CHECKPOINT_REMOTE_PATH: checkpoint_volume
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
    env["MODEL_PATH"] = os.path.join(MODEL_REMOTE_PATH, "Qwen3-4B-Instruct-2507") # Qwen3-4B-Instruct-2507 model
    env["DATA_PATH"] = os.path.join(DATA_REMOTE_PATH, "airline_sft_formatted.jsonl")
    env["OUTPUT_PATH"] = os.path.join(CHECKPOINT_REMOTE_PATH, "Qwen3-4B-sft-airline")
    env["MEGATRON_PATH"] = MEGATRON_REMOTE_PATH
    
    # Run the SFT script
    print("Launching SFT Training...")
    result = subprocess.run([sft_script_path], env=env, text=True, capture_output=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}")

    print("Training completed successfully.")

@app.function(
    gpu="H100", # Cheaper than H100 for testing
    timeout=60*10, # 10 minutes timeout
    volumes={
        DATA_REMOTE_PATH: data_volume,
        MODEL_REMOTE_PATH: model_volume,
        CHECKPOINT_REMOTE_PATH: checkpoint_volume
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
    env["DATA_PATH"] = os.path.join(DATA_REMOTE_PATH, "airline_sft_formatted.jsonl")
    env["OUTPUT_PATH"] = os.path.join(CHECKPOINT_REMOTE_PATH, "Qwen3-4B-sft-airline-test")
    env["MEGATRON_PATH"] = MEGATRON_REMOTE_PATH
    
    # Test specific overrides
    env["NUM_EPOCH"] = "1"
    env["BATCH_SIZE"] = "1"
    
    # Run the SFT script
    print("Launching SFT Test Training...")
    result = subprocess.run([sft_script_path], env=env, text=True, capture_output=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise Exception(f"Test Training failed with exit code {result.returncode}")

    print("Test Training completed successfully.")

@app.function(
    gpu="H100",
    timeout=3600, # 1 hour should be plenty for conversion
    volumes={
        MODEL_REMOTE_PATH: model_volume
    },
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
