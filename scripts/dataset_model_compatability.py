import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random

def load_squad_sample(num_samples=5):
    dataset = load_dataset("squad")
    return random.sample(list(dataset['train']), num_samples)

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def format_squad_example(example):
    return f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answers']['text'][0]}"

def check_tokenizer_compatibility(tokenizer, formatted_example):
    tokens = tokenizer.tokenize(formatted_example)
    print(f"Number of tokens: {len(tokens)}")
    print(f"Sample tokens: {tokens[:10]}...")
    
    # Check for unknown tokens
    unknown_tokens = [token for token in tokens if token == tokenizer.unk_token]
    print(f"Number of unknown tokens: {len(unknown_tokens)}")

def check_model_output(model, tokenizer, formatted_example):
    inputs = tokenizer(formatted_example, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}")

def main():
    model_path = "./llama_3_2_1b_model"
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    squad_samples = load_squad_sample()
    
    for i, example in enumerate(squad_samples, 1):
        print(f"\n--- Example {i} ---")
        formatted_example = format_squad_example(example)
        print(f"Formatted example:\n{formatted_example}\n")
        
        print("Tokenizer Compatibility:")
        check_tokenizer_compatibility(tokenizer, formatted_example)
        
        print("\nModel Output:")
        check_model_output(model, tokenizer, formatted_example)

if __name__ == "__main__":
    main()