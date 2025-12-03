
import json
from typing import List, Dict, Tuple

# Mock AutoTokenizer for testing purposes
class MockTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else {"<|im_start|>": 0, "<|im_end|>": 1, "system": 2, "user": 3, "assistant": 4, "tool": 5, "hello": 6, "world": 7, "how": 8, "are": 9, "you": 10, "i": 11, "am": 12, "fine": 13, "call_tool": 14, "tool_output": 15, "name=tool_x": 16}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_token_id = len(self.vocab)

    def _get_token_id(self, token_text):
        if token_text not in self.vocab:
            self.vocab[token_text] = self.next_token_id
            self.reverse_vocab[self.next_token_id] = token_text
            self.next_token_id += 1
        return self.vocab[token_text]

    def encode(self, text, add_special_tokens=False):
        # Very simplistic tokenizer: split by space and map to IDs
        tokens = text.split()
        encoded_tokens = [self._get_token_id(t) for t in tokens]
        return encoded_tokens

    def apply_chat_template(self, messages: List[Dict], tokenize: bool, add_generation_prompt: bool, tools=None) -> str:
        # Simplistic chat template application
        chat_string = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if role == "system":
                chat_string += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                chat_string += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                chat_string += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                name = message.get("name", "tool")
                chat_string += f"<|im_start|>tool\nname={name}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            chat_string += "<|im_start|>assistant\n"
        return chat_string

# The _get_token_delta function from trainable_agents.py
def _get_token_delta(tokenizer: MockTokenizer, messages: List[Dict]) -> Tuple[List[int], List[int]]:
    curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    token_ids = []
    loss_mask = []

    # Case 1: last message is an assistant response
    if messages[-1]["role"] == "assistant":
        prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
        new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
        token_ids += new_tokens
        loss_mask += [1] * len(new_tokens)  # Mask only the new assistant tokens
    else:
        # Case 2: last message is a tool response or environment observation
        prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)
        new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
        token_ids += new_tokens
        loss_mask += [0] * len(new_tokens)  # Don't mask environment/tool tokens

    return token_ids, loss_mask

# Test cases
def run_tests():
    tokenizer = MockTokenizer()

    # Test Case 1: Assistant response
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "How can I help you?"}
    ]
    token_ids1, loss_mask1 = _get_token_delta(tokenizer, messages1)
    print(f"Test Case 1 (Assistant): Token IDs: {token_ids1}, Loss Mask: {loss_mask1}")
    # Expected: loss_mask should be all 1s for the assistant's new response tokens

    # Test Case 2: Tool response
    messages2 = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "Call tool x"},
        {"role": "assistant", "content": "Calling tool x."}, 
        {"role": "tool", "name": "tool_x", "content": "Tool x output."}
    ]
    token_ids2, loss_mask2 = _get_token_delta(tokenizer, messages2)
    print(f"Test Case 2 (Tool): Token IDs: {token_ids2}, Loss Mask: {loss_mask2}")
    # Expected: loss_mask should be all 0s for the tool's new response tokens

    # Test Case 3: User response after assistant
    messages3 = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "I am good, how are you?"}
    ]
    token_ids3, loss_mask3 = _get_token_delta(tokenizer, messages3)
    print(f"Test Case 3 (User): Token IDs: {token_ids3}, Loss Mask: {loss_mask3}")
    # Expected: loss_mask should be all 0s for the user's new response tokens

if __name__ == "__main__":
    run_tests()
