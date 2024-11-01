# main.py

import torch
from pathlib import Path
import pytorch_lightning as pl
import wandb
import time
import curses
import logging

# Pytorch Lightning Imports
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

# Local Imports
from config import TrainingConfig
from dataloader import TextClassificationDataModule
from lightning_model import LightningClassifier # check in light
from model_utils import setup_lightning_model


def run_training():
    # Add logging setup at the start
    logging.basicConfig(
        filename='training.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize config (keep existing config setup)
    config = TrainingConfig(
        model_name="facebook/opt-125m",
        train_size=1000,
        eval_size=100,
        batch_size=12,
        num_epochs=3,
        model_max_length=512,
        learning_rate=1e-6, # consider reducing this 
        weight_decay=0.01,
        warmup_ratio=0.1,
        output_dir="./lightning_outputs",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=1000,
        save_steps=100,
        early_stopping_patience=50,
        num_train_samples=1000,  # Added
        num_eval_samples=100,     # Added
        save_top_k=3
    )

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="sentiment-analysis",
                               name=f"run-{config.model_name}",
                               config=config.__dict__)
    
    # Initialize callbacks
    callbacks = [ModelCheckpoint(dirpath=config.output_dir,
                                  filename="{epoch}-{val_f1:.2f}",
                                  monitor="val_f1",
                                  mode="max",
                                  save_top_k=config.save_top_k,
                                  save_last=True),
                 EarlyStopping(monitor="val_loss",
                               patience=config.early_stopping_patience,
                               mode="min")]


    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu' if config.device == 'cuda' else 'cpu',
                         callbacks=callbacks,
                         logger=wandb_logger,
                         max_epochs=config.num_epochs,
                         max_steps=config.max_steps,
                         precision='16-mixed' if config.device == 'cuda' else '32',
                         gradient_clip_val=1.0,
                         gradient_checkpointing=True)

    # Initialize model and data module
    model, tokenizer = setup_lightning_model(config)
    data_module = TextClassificationDataModule(config, tokenizer)

    # Train model
    try:
        trainer.fit(model, datamodule=data_module)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        wandb.finish()

if __name__ == "__main__":
    run_training()