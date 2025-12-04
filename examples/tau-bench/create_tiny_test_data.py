"""Create a tiny test dataset with just 2 conversations for quick testing."""

import json

def create_tiny_test_data(input_path, output_path, num_samples=2):
    """Copy first N conversations to a test file."""

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for i, line in enumerate(infile):
            if i >= num_samples:
                break
            outfile.write(line)

    print(f"Created test dataset with {num_samples} conversations: {output_path}")

if __name__ == "__main__":
    create_tiny_test_data(
        "examples/tau-bench/data/airline_sft_filtered.jsonl",
        "examples/tau-bench/data/airline_sft_tiny_test.jsonl",
        num_samples=2  # Just 2 conversations for quick test
    )
