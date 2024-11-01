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
