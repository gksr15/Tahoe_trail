import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
from dataset import TahoeDataset, get_dataloader
import torch.utils.tensorboard as tensorboard

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,  
        min_learning_rate: float = 7e-5,  
        lr_decay_steps: int = 3000,  
        weight_decay: float = 1e-5,
        log_every_n_steps: int = 10,
        eval_every_n_steps: int = 250,
        eval_num_batches: int = 100,
        checkpoint_dir: str = "checkpoints",
        run_name: Optional[str] = None
    ):
        # Setup checkpoint and logging directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.checkpoint_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging first
        self._setup_logging()
        
        # Log training parameters
        self.logger.info(f"Initializing trainer with:")
        self.logger.info(f"Initial learning rate: {learning_rate}")
        self.logger.info(f"Minimum learning rate: {min_learning_rate}")
        self.logger.info(f"Learning rate decay steps: {lr_decay_steps}")
        self.logger.info(f"Weight decay: {weight_decay}")
        self.logger.info(f"Evaluation frequency: Every {eval_every_n_steps} steps")
        self.logger.info(f"Evaluation batches: {eval_num_batches}")
        
        # Initialize CUDA and check GPU availability
        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Using CPU instead.")
            device = "cpu"
        else:
            # Set CUDA device
            torch.cuda.set_device(0) 
            self.logger.info(f"CUDA is available! Using device: {torch.cuda.get_device_name(0)}")
        
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        
        # Log GPU information
        if self.device == "cuda":
            self.logger.info(f"Found {self.num_gpus} GPUs!")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                self.logger.info(f"GPU {i}: {gpu_name}")
                self.logger.info(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                self.logger.info(f"Memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            
                self.model = model.to(self.device)
        
        # Initialize dataloaders as None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler()
        
        # Set up moa-broad mapping
        self.moa_broad_to_idx = {
            'activator/agonist': 0,
            'inhibitor/antagonist': 1,
            'unclear': 2
        }
        
        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=lr_decay_steps,
            eta_min=min_learning_rate
        )
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.weight_decay = weight_decay
        self.eval_num_batches = eval_num_batches
        
        # Logging setup
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # TensorBoard setup - organize by runs
        self.writer = tensorboard.SummaryWriter(log_dir=self.run_dir / 'tensorboard')
        
    def _setup_logging(self):
        
        log_file = self.run_dir / f"training_{self.run_name}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate classification accuracy"""
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()
            total = labels.size(0)
            return correct / total
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device efficiently
        gene_exp = batch['gene_exp_norm'].to(self.device, non_blocking=True)
        # Convert moa-broad strings to indices
        labels = torch.tensor([self.moa_broad_to_idx[moa] for moa in batch['moa_broad']], device=self.device)
        
        # Forward pass with autocast for mixed precision
        self.optimizer.zero_grad()
        
        # Use autocast for mixed precision training
        with torch.amp.autocast(device_type=self.device):
            logits = self.model(gene_exp)
            loss = self.criterion(logits, labels)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping before optimizer step
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step and update scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate accuracy
        with torch.no_grad():
            accuracy = self._calculate_accuracy(logits.detach(), labels)
        
        # Store current metrics
        self.current_train_loss = loss.item()
        self.current_train_accuracy = accuracy
        
        return loss.item(), accuracy
    
    def _evaluate(self, loader: DataLoader, prefix: str = "val", num_batches: Optional[int] = None) -> Tuple[float, float]:
        """
        Evaluate model on given dataloader
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        batch_count = 0
        
        # Default to evaluating on 25% of the data if num_batches not specified
        if num_batches is None:
            num_batches = max(len(loader) // 4, 10)  # At least 10 batches
        
        with torch.no_grad():
            for batch in tqdm(loader, total=num_batches, desc=f"{prefix.capitalize()} Evaluation"):
                if batch_count >= num_batches:
                    break
                    
                gene_exp = batch['gene_exp_norm'].to(self.device, non_blocking=True)
                # Convert moa-broad strings to indices
                labels = torch.tensor([self.moa_broad_to_idx[moa] for moa in batch['moa_broad']], device=self.device)
                
                # Use autocast for mixed precision
                with torch.amp.autocast(device_type=self.device):
                    logits = self.model(gene_exp)
                    loss = self.criterion(logits, labels)
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_accuracy += self._calculate_accuracy(logits, labels) * batch_size
                total_samples += batch_size
                batch_count += 1
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        
        # Log metrics to TensorBoard with appropriate prefix
        self.writer.add_scalar(f"{prefix}/loss", avg_loss, self.global_step)
        self.writer.add_scalar(f"{prefix}/accuracy", avg_accuracy, self.global_step)
        
        return avg_loss, avg_accuracy

    def _save_checkpoint(self, checkpoint_name: str = None):
        """Save model checkpoint        """
        # Prepare metrics dictionary
        metrics = {
            'train_loss': getattr(self, 'current_train_loss', None),
            'train_accuracy': getattr(self, 'current_train_accuracy', None),
            'val_loss': getattr(self, 'current_val_loss', None),
            'val_accuracy': getattr(self, 'current_val_accuracy', None),
            'test_loss': getattr(self, 'current_test_loss', None),
            'test_accuracy': getattr(self, 'current_test_accuracy', None),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # Added scheduler state
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics
        }
        
        if checkpoint_name:
            checkpoint_path = self.checkpoint_dir / f'{checkpoint_name}.pt'
            torch.save(checkpoint, checkpoint_path)
            return
            
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pt')
        
        # Save periodic checkpoint
        if self.global_step % 1000 == 0:  # Save every 1000 steps
            periodic_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'
            torch.save(checkpoint, periodic_path)
            
            # Log current metrics
            self.logger.info(f"\nStep {self.global_step} Metrics:")
            self.logger.info(f"Train - Loss = {metrics['train_loss']:.4f}, Accuracy = {metrics['train_accuracy']:.4f}")
            if metrics['val_loss'] is not None:
                self.logger.info(f"Val - Loss = {metrics['val_loss']:.4f}, Accuracy = {metrics['val_accuracy']:.4f}")
            if metrics['test_loss'] is not None:
                self.logger.info(f"Test - Loss = {metrics['test_loss']:.4f}, Accuracy = {metrics['test_accuracy']:.4f}")

    def _compute_class_distribution(self, loader: DataLoader) -> torch.Tensor:
        """Compute distribution of MOA broad classes in the dataset"""
        moa_counts = torch.zeros(3)  
        
        for batch in loader:
            moa_indices = [self.moa_broad_to_idx[moa] for moa in batch['moa_broad']]
            
            # Count occurrences of each class
            for idx in moa_indices:
                moa_counts[idx] += 1
                
        return moa_counts

    def train(self, num_epochs: int):
        """Main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Fixed learning rate: {self.learning_rate}")
        
        # Log GPU information
        if self.device == "cuda":
            self.logger.info(f"Using {self.num_gpus} GPU(s)")
            for i in range(self.num_gpus):
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                self.logger.info(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                self.logger.info(f"Memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training loop
            train_losses = []
            train_accuracies = []
            running_loss = 0.0
            running_acc = 0.0
            num_samples = 0
            
            self.model.train()
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
                loss, accuracy = self._train_step(batch)
                
                # Update running statistics
                batch_size = len(batch['moa_broad'])
                running_loss += loss * batch_size
                running_acc += accuracy * batch_size
                num_samples += batch_size
                
                # Keep track for epoch statistics
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                
                self.global_step += 1
                
                # Calculate running averages
                avg_loss = running_loss / num_samples
                avg_acc = running_acc / num_samples
                
                # Log training metrics to TensorBoard (smoothed)
                self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                self.writer.add_scalar("train/accuracy", avg_acc, self.global_step)
                self.writer.add_scalar("train/learning_rate", self.learning_rate, self.global_step)
                
                # Log training metrics to console
                if self.global_step % self.log_every_n_steps == 0:
                    self.logger.info(
                        f"Step {self.global_step}: "
                        f"Train Loss = {avg_loss:.4f}, "
                        f"Train Acc = {avg_acc:.4f}, "
                        f"LR = {self.learning_rate:.6f}"
                    )
                
                # Save checkpoint every 500 steps
                if self.global_step % 500 == 0:
                    self.logger.info(f"\nSaving checkpoint at step {self.global_step}")
                    # Store current training metrics
                    self.current_train_loss = avg_loss
                    self.current_train_accuracy = avg_acc
                    self._save_checkpoint(checkpoint_name=f'checkpoint_step_{self.global_step}')
            
            # End of epoch evaluation
            self.logger.info("\nRunning full evaluation at end of epoch...")
            
            # Calculate final epoch metrics
            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = np.mean(train_accuracies)
            
            # Full evaluation at end of epoch
            val_loss, val_acc = self._evaluate(self.val_loader, prefix="val", num_batches=None)  
            test_loss, test_acc = self._evaluate(self.test_loader, prefix="test", num_batches=None) 
            
            # Store current metrics
            self.current_train_loss = epoch_train_loss
            self.current_train_accuracy = epoch_train_acc
            self.current_val_loss = val_loss
            self.current_val_accuracy = val_acc
            self.current_test_loss = test_loss
            self.current_test_accuracy = test_acc
            
            self.logger.info(
                f"Epoch {epoch + 1} Results:\n"
                f"Train - Loss = {epoch_train_loss:.4f}, Accuracy = {epoch_train_acc:.4f}\n"
                f"Val - Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}\n"
                f"Test - Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}"
            )
            
            # Save epoch checkpoint
            self._save_checkpoint(checkpoint_name=f'checkpoint_epoch_{epoch + 1}')
            
            # Compute and log class distributions
            moa_dist = self._compute_class_distribution(self.train_loader)
            total_samples = moa_dist.sum().item()
            
            self.logger.info(f"\nEpoch {epoch + 1} Statistics:")
            self.logger.info(f"Total training samples: {total_samples}")
            self.logger.info("\nMOA Broad class distribution:")
            for moa, idx in self.moa_broad_to_idx.items():
                count = moa_dist[idx].item()
                percentage = (count / total_samples) * 100
                self.logger.info(f"{moa}: {count} samples ({percentage:.2f}%)")
                # Log class distribution to TensorBoard
                self.writer.add_scalar(f"class_distribution/{moa}", percentage, epoch)
        
        # Close TensorBoard writer
        self.writer.close()

    def create_data_loader(
        self,
        dataset: TahoeDataset,
        batch_size: int = 32,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        num_workers: int = 0,  
        seed: int = 42,
        max_val_samples: Optional[int] = 5000,
        max_test_samples: Optional[int] = 5000,
        persistent_workers: bool = False,  
        pin_memory: bool = True, 
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders from dataset with fixed seeds and size limits"""
        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1")
            
        # Calculate split sizes
        total_size = len(dataset)
        
        if max_val_samples:
            val_size = min(int(val_ratio * total_size), max_val_samples)
        else:
            val_size = int(val_ratio * total_size)
            
        if max_test_samples:
            test_size = min(int(test_ratio * total_size), max_test_samples)
        else:
            test_size = int(test_ratio * total_size)
        
        train_size = total_size - val_size - test_size
        
        # Set random seed for reproducibility
        generator = torch.Generator().manual_seed(seed)
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        self.logger.info(
            f"Dataset splits (after size limits):\n"
            f"Train={train_size} ({train_size/total_size:.1%})\n"
            f"Val={val_size} ({val_size/total_size:.1%})\n"
            f"Test={test_size} ({test_size/total_size:.1%})"
        )
        
        # Create dataloaders with optimized settings
        train_loader = get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )
        
        # Use larger batch size for validation and test for faster evaluation
        eval_batch_size = batch_size * 2
        
        val_loader = get_dataloader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )
        
        test_loader = get_dataloader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
 
    from model import create_model
    
    # Create dataset
    dataset = TahoeDataset(
        data_source="hf",
        split="train",
        max_samples=500000,  
        gene_max_cnt=1000
    )
    
    model = create_model(
        input_dim=1000,
        d_model=64,
        nhead=4,
        num_layers=4,
        num_classes=3
    )

    trainer = Trainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-5,
        log_every_n_steps=10,
        eval_every_n_steps=250, 
        eval_num_batches=100  # Evaluate on 3200 samples (100 * 32)
    )

    train_loader, val_loader, test_loader = trainer.create_data_loader(
        dataset,
        batch_size=32,
        num_workers=0,  
        pin_memory=True,  
        persistent_workers=False  
    )

    trainer.train_loader = train_loader
    trainer.val_loader = val_loader
    trainer.test_loader = test_loader

    trainer.train(num_epochs=5)
