commit 1: Initial Migration Setup
- Remove dashboard.py and monitor.py (replaced by W&B)
- Add requirements.txt with pytorch-lightning and wandb dependencies

commit 2: Add Lightning Model
- Create lightning_model.py with LightningClassifier
- Migrate training logic from trainer.py [startLine: 155, endLine: 227]
- Migrate evaluation logic [startLine: 74, endLine: 115]
- Migrate optimization setup [startLine: 52, endLine: 72]

commit 3: Add Lightning DataModule
- Create new dataloader.py with TextClassificationDataModule
- Migrate dataset handling and collate functions
- Add proper data splitting for train/val

commit 4: Update Configuration
- Update config.py with Lightning-specific parameters
- Add save_top_k parameter [startLine: 23]
- Add W&B project configuration options

commit 5: Refactor Main Training Loop
- Update main.py to use Lightning Trainer
- Add W&B logger initialization [startLine: 57, endLine: 59]
- Add Lightning callbacks for checkpointing and early stopping [startLine: 62, endLine: 70]
- Remove old training initialization code

commit 6: Migrate Metrics Logging
- Move metric logging to W&B
- Add confusion matrix visualization
- Add gradient tracking
- Remove old metrics storage

commit 7: Cleanup Legacy Code
- Remove trainer.py (replaced by Lightning Trainer)
- Remove results.md (replaced by W&B artifacts)
- Clean up imports in main.py

commit 8: Add Documentation
- Add README.md with migration notes
- Document W&B setup process
- Add example configuration


commit: Migrate to PyTorch Lightning with W&B Integration

BREAKING CHANGE: Complete architecture overhaul

feat(core):
- Replace custom training loop with PyTorch Lightning
- Integrate Weights & Biases for experiment tracking
- Add LoRA support in Lightning framework

refactor:
- Remove dashboard.py, monitor.py, and trainer.py
- Migrate model initialization to lightning_model_utils.py
- Update dataloader to Lightning DataModule
- Convert inference to Lightning-compatible structure

changes:
1. lightning_model.py
   - Add LightningModule implementation
   - Configure optimizers with AdamW and warmup
   - Implement training and validation steps
   Ref: startLine: 8, endLine: 91

2. main.py
   - Remove curses-based UI
   - Add W&B logger configuration
   - Setup Lightning trainer with callbacks
   - Initialize model and data module
   Ref: startLine: 27, endLine: 93

3. config.py
   - Add Lightning-specific parameters
   - Include W&B configuration
   - Add LoRA parameters
   Ref: startLine: 7, endLine: 46

4. lightning_model_utils.py
   - Add Lightning model setup
   - Integrate LoRA configuration
   - Handle tokenizer initialization
   Ref: startLine: 8, endLine: 40

5. lightning_inference.py
   - Add W&B logging for predictions
   - Implement batch prediction
   - Add test model functionality
   Ref: startLine: 7, endLine: 94

dependencies:
+ pytorch-lightning>=2.0.0
+ wandb
+ transformers
+ peft
+ datasets
+ scikit-learn
+ torch>=2.0.0

Closes #123
