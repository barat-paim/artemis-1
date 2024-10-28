import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_answer(text):
    """Clean up answer text for better matching"""
    # Remove explanations and extra context
    text = text.split('\n')[0].split('Explanation:')[0]
    # Remove common prefixes
    text = text.replace('The answer is', '').replace('Answer:', '')
    return text.strip().lower()

def compute_exact_match(prediction, truth):
    prediction = clean_answer(prediction)
    truth = clean_answer(truth)
    return int(prediction == truth)

def compute_f1(prediction, truth):
    prediction = clean_answer(prediction)
    truth = clean_answer(truth)
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if not common_tokens:
        return 0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(model, tokenizer, eval_dataset, num_examples=None, save_results=True):
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    total_f1 = 0
    total_em = 0
    
    # Take subset if specified
    if num_examples:
        eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))
    
    logger.info(f"Evaluating on {len(eval_dataset)} examples...")
    
    for idx in tqdm(range(len(eval_dataset))):
        example = eval_dataset[idx]
        
        # Prepare input
        input_text = (
            "Answer the question briefly using only the provided context. "
            "Give a short, direct answer without explanations.\n\n"
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            "Answer:"
        )
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=4,
                temperature=0.3,
                top_p=0.9,
                no_repeat_ngram_size=3,
                length_penalty=0.6,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Clean up the generated answer
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated_answer.replace(input_text, "").strip()
        # Remove any "Explanation:" or additional context
        generated_answer = generated_answer.split('\n')[0].split('Explanation:')[0].strip()
        true_answer = example['answers']['text'][0]
        
        # Calculate metrics
        f1_score = compute_f1(generated_answer, true_answer)
        em_score = compute_exact_match(generated_answer, true_answer)
        
        total_f1 += f1_score
        total_em += em_score
        
        # Store result
        result = {
            'question': example['question'],
            'context': example['context'],
            'generated_answer': generated_answer,
            'true_answer': true_answer,
            'f1_score': f1_score,
            'em_score': em_score
        }
        results.append(result)
        
        # Log every 10 examples
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1} examples. Current Avg F1: {total_f1/(idx+1):.4f}, EM: {total_em/(idx+1):.4f}")
    
    # Calculate final metrics
    avg_f1 = total_f1 / len(eval_dataset)
    avg_em = total_em / len(eval_dataset)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_results_small_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'metrics': {
                    'avg_f1': avg_f1,
                    'avg_em': avg_em
                },
                'results': results
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return avg_f1, avg_em, results

def main():
    # Load model and tokenizer
    model_path = "./fine_tuned_llama_squad_improved"
    logger.info("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load evaluation dataset
    logger.info("Loading evaluation dataset...")
    squad_dataset = load_from_disk("./data/squad")
    eval_dataset = squad_dataset['validation']
    
    # Evaluate on a small subset first
    num_eval_examples = 50  # Start with a small number
    avg_f1, avg_em, results = evaluate_model(
        model, 
        tokenizer, 
        eval_dataset, 
        num_examples=num_eval_examples
    )
    
    logger.info("\nEvaluation Results:")
    logger.info(f"Average F1 Score: {avg_f1:.4f}")
    logger.info(f"Average Exact Match Score: {avg_em:.4f}")
    
    # Print a few example predictions
    logger.info("\nSample Predictions:")
    for i, result in enumerate(results[:5]):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Question: {result['question']}")
        logger.info(f"Generated Answer: {result['generated_answer']}")
        logger.info(f"True Answer: {result['true_answer']}")
        logger.info(f"F1 Score: {result['f1_score']:.4f}")
        logger.info(f"EM Score: {result['em_score']}")

if __name__ == "__main__":
    main()
