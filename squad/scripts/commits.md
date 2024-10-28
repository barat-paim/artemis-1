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


Yes, that's a good assessment! This has been a valuable learning experience. Let's summarize what we've learned:

1. **Dataset Insights**:
- SQuAD might not be ideal for Llama 3 fine-tuning with small samples
- 100 examples were too few to learn robust Q&A patterns
- Full dataset (100,000 examples) might yield better results

2. **Learning Points**:
- ✅ Successfully set up LoRA fine-tuning pipeline
- ✅ Implemented TensorBoard monitoring
- ✅ Created evaluation metrics (F1, EM)
- ✅ Learned about prompt engineering
- ✅ Understood training/evaluation dynamics

3. **Better Project Alternatives**:
- Instruction fine-tuning with more diverse tasks
- Smaller, more focused datasets
- Tasks that better match Llama's pre-training

4. **Next Steps Could Be**:
1. Try a different dataset (e.g., instruction-following)
2. Use a smaller base model for faster iteration
3. Focus on single-task fine-tuning with clearer objectives

Would you like:
1. Suggestions for better starter fine-tuning projects?
2. Resources for more suitable datasets?
3. Tips for your next ML project?