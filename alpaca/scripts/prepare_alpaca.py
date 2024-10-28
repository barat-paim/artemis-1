import datasets
from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm
import logging
import json
from typing import Dict, List
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_filter_dataset(num_examples: int = 2000) -> datasets.Dataset:
    """Load and filter the Alpaca dataset."""
    logger.info("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")
    
    def filter_example(example: Dict) -> bool:
        """Filter criteria for examples."""
        # Check lengths
        if len(example['instruction']) > 200 or len(example['output']) > 200:
            return False
            
        # Avoid complex or ambiguous tasks
        skip_phrases = ['code', 'program', 'write a story', 'essay']
        if any(phrase in example['instruction'].lower() for phrase in skip_phrases):
            return False
            
        # Ensure input is not too long or empty
        if example['input'] and len(example['input']) > 100:
            return False
            
        return True
    
    # Filter dataset
    filtered = dataset['train'].filter(filter_example)
    
    # Select categories we want to focus on
    task_categories = {
        'text_transformation': [
            'convert', 'transform', 'change', 'rewrite', 'rephrase'
        ],
        'qa': [
            'answer', 'what is', 'how do', 'explain'
        ],
        'grammar': [
            'correct', 'fix', 'improve', 'grammar'
        ],
        'summarization': [
            'summarize', 'summarise', 'brief', 'shorten'
        ]
    }
    
    def get_category(example: Dict) -> str:
        """Determine the category of an instruction."""
        instruction = example['instruction'].lower()
        for category, keywords in task_categories.items():
            if any(keyword in instruction for keyword in keywords):
                return category
        return 'other'
    
    # Add categories and select balanced subset
    logger.info("Categorizing and balancing dataset...")
    examples = []
    category_counts = {cat: 0 for cat in task_categories.keys()}
    per_category = num_examples // len(task_categories)
    
    for example in tqdm(filtered):
        category = get_category(example)
        if category in category_counts and category_counts[category] < per_category:
            example['category'] = category
            examples.append(example)
            category_counts[category] += 1
    
    # Shuffle and limit to desired size
    random.shuffle(examples)
    examples = examples[:num_examples]
    
    # Convert to dataset
    prepared_dataset = datasets.Dataset.from_pandas(pd.DataFrame(examples))
    
    # Log statistics
    logger.info(f"Dataset statistics:")
    for cat, count in category_counts.items():
        logger.info(f"{cat}: {count} examples")
    
    return prepared_dataset

def format_instruction(example: Dict) -> Dict:
    """Format the instruction template."""
    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n"
    )
    
    if example['input']:
        template += f"### Input:\n{example['input']}\n"
    
    template += "### Response:\n"
    
    return {
        'text': template,
        'instruction': example['instruction'],
        'input': example['input'],
        'output': example['output'],
        'category': example['category']
    }

def save_dataset(dataset: datasets.Dataset, output_dir: str = "./data"):
    """Save the prepared dataset."""
    # Save full dataset
    dataset.save_to_disk(f"{output_dir}/alpaca_prepared")
    
    # Save train/val split
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    splits['train'].save_to_disk(f"{output_dir}/alpaca_prepared/train")
    splits['test'].save_to_disk(f"{output_dir}/alpaca_prepared/validation")
    
    # Save sample as JSON for inspection
    sample = dataset.select(range(5))
    with open(f"{output_dir}/alpaca_sample.json", 'w') as f:
        json.dump(sample.to_dict(), f, indent=2)

def main():
    # Prepare dataset
    dataset = load_and_filter_dataset(num_examples=2000)
    
    # Format instructions
    logger.info("Formatting instructions...")
    dataset = dataset.map(format_instruction)
    
    # Save dataset
    logger.info("Saving dataset...")
    save_dataset(dataset)
    
    logger.info("Dataset preparation complete!")
    logger.info(f"Total examples: {len(dataset)}")
    
    # Show a sample
    logger.info("\nSample instruction:")
    sample = dataset[0]
    logger.info(f"\nCategory: {sample['category']}")
    logger.info(f"Instruction: {sample['instruction']}")
    logger.info(f"Input: {sample['input']}")
    logger.info(f"Output: {sample['output']}")

if __name__ == "__main__":
    main()