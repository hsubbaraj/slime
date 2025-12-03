
import json
from datasets import load_dataset

def test_loading():
    data_path = "slime/examples/tau-bench/data/airline_sft_formatted.jsonl"
    print(f"Loading dataset from {data_path}...")
    
    # Try loading with Hugging Face datasets
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
        print(f"Successfully loaded dataset. Size: {len(dataset)}")
        print("First sample:")
        print(dataset[0])
        
        # Verify structure
        sample = dataset[0]
        if "messages" not in sample:
            print("Error: 'messages' key missing in dataset.")
        else:
            print("Structure check passed: 'messages' key present.")
            
    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    test_loading()
