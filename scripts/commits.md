training: create finetune.py for small-scale training experiment

- Reduce training set to 100 examples
- Add comprehensive memory and speed tracking
- Limit training to 2 epochs
- Reduce LoRA parameters for lighter training
- Implement detailed logging for analysis

feat: Add evaluation script for small fine-tuned model

- Implement detailed evaluation metrics (F1 and EM scores)
- Add progress tracking and logging
- Save results to JSON for analysis
- Include sample prediction display
- Support for subset evaluation

docs: Analyze evaluation results of small fine-tuned model

- Document poor performance metrics (F1: 6.48%, EM: 0%)
- Identify issues with answer formatting and context repetition
- Suggest improvements for training data and model architecture

feat: Implement improved finetune-v2 with better prompts and data formatting

- Add ImprovedSQuADDataset with clearer prompt structure
- Enhance LoRA configuration with additional target modules
- Adjust training parameters for better optimization
- Implement explicit answer boundaries and length constraints

feat: Add TensorBoard logging to improved fine-tuning

- Implement TensorBoard metrics tracking
- Add custom callback for detailed metric logging
- Track training/eval losses, learning rate, and GPU usage
- Enable real-time visualization of training progress
- Add proper cleanup of TensorBoard writer